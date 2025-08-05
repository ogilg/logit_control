#!/usr/bin/env python3
"""
LoRA fine-tuning script for qwen2-3b on output control task using custom_evals.
"""

import argparse
import os
import random
from typing import List

import torch

from provider_wrapper import (
    LoRAConfig, 
    LoRAFineTuner, 
    TVDLoss, 
    KLDivergenceLoss,
    HUGGINGFACE_MODEL_MAPPING
)
from utils import (
    get_given_words_samples,
    _generate_random_words_samples,
    get_combined_samples
)


def create_training_samples(
    num_given_examples: int = 50,
    num_random_examples: int = 50,
    num_icl_examples: int = 50
) -> List:
    """Create training samples using the output control data generation functions."""
    
    # Generate samples for each case type
    given_samples = get_given_words_samples(num_given_examples)
    random_samples = _generate_random_words_samples(num_random_examples, use_few_shot=False)
    icl_samples = _generate_random_words_samples(num_icl_examples, use_few_shot=True)
    
    # Combine all samples
    all_samples = given_samples + random_samples + icl_samples
    
    print(f"Created {len(all_samples)} training samples:")
    print(f"  - Given words: {len(given_samples)}")
    print(f"  - Random words: {len(random_samples)}")
    print(f"  - ICL random words: {len(icl_samples)}")
    
    return all_samples


def get_model_output_dir(model_name: str) -> str:
    """Create a model-specific output directory name."""
    # Clean the model name for use as directory name
    clean_name = model_name.replace("/", "_").replace("-", "_").replace(".", "_")
    return f"./lora_output/lora_output_{clean_name}"


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for output control task using custom_evals")
    
    # Model and data parameters
    parser.add_argument("--model_name", type=str, default="qwen2.5-3b-instruct", 
                       help="Model name to fine-tune")
    parser.add_argument("--num_given_examples", type=int, default=50,
                       help="Number of given words examples")
    parser.add_argument("--num_random_examples", type=int, default=5,
                       help="Number of random words examples")
    parser.add_argument("--num_icl_examples", type=int, default=5,
                       help="Number of ICL random words examples")
    
    # LoRA configuration
    parser.add_argument("--lora_r", type=int, default=1,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                       help="LoRA alpha")
    parser.add_argument("--lora_target", type=str, default="both", 
                       choices=["attention", "mlp", "both"],
                       help="LoRA target layers")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout")
    
    # Loss function
    parser.add_argument("--loss_type", type=str, default="kl",
                       choices=["tvd", "kl"],
                       help="Loss function type")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Warmup steps")

    parser.add_argument("--hub_model_id", type=str, 
                       help="HuggingFace hub model ID for saving adapter")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    output_dir = get_model_output_dir(args.model_name)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting LoRA fine-tuning on {args.model_name}")
    print(f"Training examples: {args.num_given_examples} given, {args.num_random_examples} random, {args.num_icl_examples} ICL")
    print(f"LoRA config: r={args.lora_r}, alpha={args.lora_alpha}, target={args.lora_target}")
    print(f"Loss function: {args.loss_type}")
    print(f"Output directory: {output_dir}")
    
    # Check if model is supported
    if args.model_name not in HUGGINGFACE_MODEL_MAPPING:
        raise ValueError(f"Model {args.model_name} not supported by HuggingFace provider")
    
    # Create training samples
    print("Creating training samples...")
    samples = create_training_samples(
        num_given_examples=args.num_given_examples,
        num_random_examples=args.num_random_examples,
        num_icl_examples=args.num_icl_examples
    )
    
    # Setup LoRA configuration
    lora_config = LoRAConfig(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target=args.lora_target,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        gradient_checkpointing=True
    )
    
    # Setup loss function
    if args.loss_type == "tvd":
        loss_function = TVDLoss()
    elif args.loss_type == "kl":
        loss_function = KLDivergenceLoss()
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")
    
    # Create fine-tuner
    print("Setting up LoRA fine-tuner...")
    finetuner = LoRAFineTuner(
        model_id=f"huggingface/{args.model_name}",
        lora_config=lora_config,
        custom_loss=loss_function
    )
    
    # Setup model
    finetuner.setup_model()
    
    # Train the model
    print("Starting training...")
    trainer = finetuner.train(
        samples=samples,
        output_dir=output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        save_steps=500,
        eval_steps=500,
        logging_steps=10
    )
    
    # Push to HuggingFace Hub if specified
    if args.hub_model_id:
        print(f"Pushing adapter to HuggingFace Hub: {args.hub_model_id}")
        finetuner.peft_model.push_to_hub(args.hub_model_id)
        print(f"Adapter saved to: https://huggingface.co/{args.hub_model_id}")
    
    print("Training completed!")


if __name__ == "__main__":
    main() 