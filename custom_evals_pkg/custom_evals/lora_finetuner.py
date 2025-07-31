"""
LoRA fine-tuning manager for custom-evals package.

This module provides functionality for fine-tuning models using LoRA with support
for custom loss functions and configurable target layers.
"""

from typing import Any, Dict, List, Optional

import torch
from dataclasses import dataclass
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)

from .huggingface_provider import HUGGINGFACE_MODEL_MAPPING
from .lora_config import LoRAConfig
from .loss_functions import CustomLossFunction, TVDLoss


@dataclass
class CustomDataCollatorWithMetadata:
    """Custom data collator that separates model inputs from metadata."""
    
    tokenizer: any
    mlm: bool = False  # For causal LM
    
    def __call__(self, features):
        # Step 1: Separate model inputs from metadata
        model_inputs = []
        metadata_batch = {
            'target_distributions': [],
            'contexts': []
        }
        
        for feature in features:
            # Extract clean model inputs
            model_input = {
                'input_ids': feature['input_ids'],
                'attention_mask': feature['attention_mask'],
                'labels': feature['labels']
            }
            model_inputs.append(model_input)
            
            # Extract metadata
            metadata_batch['target_distributions'].append(feature.get('target_distribution', {}))
            metadata_batch['contexts'].append(feature.get('context', {}))
        
        # Step 2: Use standard collator for model inputs only
        standard_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=self.mlm
        )
        batch = standard_collator(model_inputs)
        
        # Step 3: Add metadata back to batch
        batch['metadata'] = metadata_batch
        
        return batch


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
            resume_download=True,
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
                "is_given_words": sample.is_given_words,
                "case_type": sample.case_type
            }
            
            # Tokenize the prompt with consistent padding
            inputs = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=512,  # Adjust as needed
                padding="max_length",  # Pad to max_length for consistency
                return_tensors="pt",
            )
            
            # Store sample data with labels for causal LM training
            dataset.append({
                "input_ids": inputs["input_ids"][0],
                "attention_mask": inputs["attention_mask"][0],
                "labels": inputs["input_ids"][0].clone(),  # Use .clone() to avoid reference issues
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
        data_collator = CustomDataCollatorWithMetadata(
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
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            # Add these lines to reduce file saving:
            save_strategy="no",  # Don't save checkpoints during training
            save_total_limit=None,  # No limit on saves (but we're not saving anyway)
            metric_for_best_model=None,  # No metric for best model
            greater_is_better=None,  # No comparison needed
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
        self.processing_class = tokenizer
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute custom loss during training."""
        # Extract metadata from the batch
        metadata = inputs.pop("metadata", {})
        target_distributions = metadata.get('target_distributions', [])
        contexts = metadata.get('contexts', [])
        
        outputs = model(**inputs)
        logits = outputs.logits
        
        if target_distributions:
            batch_size = logits.shape[0]
            total_loss = 0
            
            # Process each sample in the batch individually
            for i in range(batch_size):
                sample_logits = logits[i:i+1]  # Keep batch dimension
                sample_target = target_distributions[i] if i < len(target_distributions) else {}
                sample_context = contexts[i] if i < len(contexts) else {}
                
                sample_loss = self.custom_loss.compute_loss(
                    sample_logits, sample_target, self.processing_class, context=sample_context
                )
                total_loss += sample_loss
                
            loss = total_loss / batch_size
        else:
            loss = outputs.loss if hasattr(outputs, 'loss') else torch.tensor(0.0)
        
        return (loss, outputs) if return_outputs else loss

