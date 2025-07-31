"""
LoRA fine-tuning manager for custom-evals package.

This module provides functionality for fine-tuning models using LoRA with support
for custom loss functions and configurable target layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType


from .lora_config import LoRAConfig
from .loss_functions import CustomLossFunction, TVDLoss, KLDivergenceLoss
from ..custom_evals.huggingface_provider import HUGGINGFACE_MODEL_MAPPING








class LoRAFineTuner:
    """Manager for LoRA fine-tuning with custom loss functions."""
    
    def __init__(self, model_id: str, lora_config: LoRAConfig, 
                 custom_loss: Optional[CustomLossFunction] = None):
        """
        Initialize the LoRA fine-tuner.
        
        Args:
            model_id: Model identifier
            lora_config: LoRA configuration
            custom_loss: Custom loss function (if None, uses standard cross-entropy)
        """
        self.model_id = model_id
        self.lora_config = lora_config
        self.custom_loss = custom_loss or TVDLoss()
        
        # Get HuggingFace model name
        if model_id.startswith("huggingface/"):
            self.hf_model_name = model_id.split("/")[1]
        else:
            self.hf_model_name = model_id
            
        if self.hf_model_name not in HUGGINGFACE_MODEL_MAPPING:
            raise ValueError(f"Model {model_id} not supported")
            
        self.hf_model_name = HUGGINGFACE_MODEL_MAPPING[self.hf_model_name]
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
    def setup_model(self):
        """Set up the model with LoRA configuration."""
        print(f"Loading model: {self.hf_model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Configure LoRA
        lora_config_dict = self.lora_config.to_dict()
        lora_config = LoraConfig(
            **lora_config_dict,
            inference_mode=False,
        )
        
        # Apply LoRA to model
        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()
        
        return self.peft_model
    
    def prepare_dataset(self, samples: List) -> List[Dict[str, Any]]:
        """Prepare dataset for training."""
        dataset = []
        
        for sample in samples:
            # Convert prompt list to formatted string
            if hasattr(sample.prompt, '__iter__') and not isinstance(sample.prompt, str):
                # It's a list of Prompt objects
                prompt_text = "\n\n".join([msg.content for msg in sample.prompt])
            else:
                # It's already a string
                prompt_text = sample.prompt
            
            # Create context with sample information
            context = {
                "word1": sample.word1,
                "word2": sample.word2,
                "tvd_tolerance": sample.tvd_tolerance,
                "is_given_words": sample.is_given_words,
                "seed": sample.seed,
                "case_type": sample.case_type
            }
            
            # Tokenize the prompt
            inputs = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=512,  # Adjust as needed
                return_tensors="pt"
            )
            
            # Store sample data
            dataset.append({
                "input_ids": inputs["input_ids"][0],
                "attention_mask": inputs["attention_mask"][0],
                "target_distribution": sample.target_distribution,
                "context": context
            })
        
        return dataset
    
    # def custom_loss_wrapper(self, model_outputs, labels, target_distributions):
    #     """Wrapper for custom loss computation."""
    #     logits = model_outputs.logits
        
    #     total_loss = 0
    #     batch_size = logits.shape[0]
        
    #     for i in range(batch_size):
    #         # Get the last token's logits for each sample
    #         last_token_logits = logits[i, -1, :]
    #         target_dist = target_distributions[i]
            
    #         # Compute custom loss
    #         loss = self.custom_loss.compute_loss(
    #             last_token_logits.unsqueeze(0), 
    #             target_dist, 
    #             self.tokenizer
    #         )
    #         total_loss += loss
        
    #     return total_loss / batch_size
    
    def train(self, samples: List, 
              output_dir: str = "./lora_output",
              num_epochs: int = 3,
              batch_size: int = 4,
              gradient_accumulation_steps: int = 4,
              warmup_steps: int = 100,
              save_steps: int = 500,
              eval_steps: int = 500,
              logging_steps: int = 10):
        """
        Train the model using LoRA fine-tuning.
        
        Args:
            samples: Training samples
            output_dir: Directory to save the fine-tuned model
            num_epochs: Number of training epochs
            batch_size: Training batch size
            gradient_accumulation_steps: Gradient accumulation steps
            warmup_steps: Number of warmup steps
            save_steps: Save model every N steps
            eval_steps: Evaluate model every N steps
            logging_steps: Log training info every N steps
        """
        
        # Set up model
        if self.peft_model is None:
            self.setup_model()
        
        # Prepare dataset
        dataset = self.prepare_dataset(samples)
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            logging_steps=logging_steps,
            learning_rate=self.lora_config.learning_rate,
            weight_decay=self.lora_config.weight_decay,
            gradient_checkpointing=self.lora_config.gradient_checkpointing,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            **self.lora_config.get_training_args()
        )
        
        # Create trainer (custom if custom loss provided, standard otherwise)
        if self.custom_loss is not None:
            trainer = CustomTrainer(
                custom_loss=self.custom_loss,
                tokenizer=self.tokenizer,
                model=self.peft_model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
            )
        else:
            trainer = Trainer(
                model=self.peft_model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
            )
        
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Training completed. Model saved to {output_dir}")
        
        return trainer
    
    def load_finetuned_model(self, model_path: str):
        """Load a fine-tuned LoRA model."""
        from peft import PeftModel
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load LoRA weights
        self.peft_model = PeftModel.from_pretrained(self.model, model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        return self.peft_model


class CustomTrainer(Trainer):
    """Simple custom trainer that handles custom loss computation."""
    
    def __init__(self, custom_loss: CustomLossFunction, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_loss = custom_loss
        self.tokenizer = tokenizer
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute custom loss during training."""
        # Get model outputs
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get target distribution and context from inputs
        target_distribution = inputs.get("target_distribution", {})
        context = inputs.get("context", {})
        
        if target_distribution:
            # Compute custom loss with context
            loss = self.custom_loss.compute_loss(logits, target_distribution, self.tokenizer, context=context)
        else:
            # Fallback to standard loss
            loss = outputs.loss if hasattr(outputs, 'loss') else torch.tensor(0.0)
        
        return (loss, outputs) if return_outputs else loss

