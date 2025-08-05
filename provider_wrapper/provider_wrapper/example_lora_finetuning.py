#!/usr/bin/env python3
"""
Example script demonstrating LoRA fine-tuning with custom loss functions.

This script shows how to fine-tune a model to match target distributions
using the custom-evals LoRA functionality.
"""

import os
from typing import Dict, List

from provider_wrapper import (KLDivergenceLoss, LoRAFineTuner, Message, Sample,
                          TVDLoss)


def create_training_samples() -> List[Sample]:
    """Create example training samples for fine-tuning."""
    
    samples = []
    
    # Example 1: Simple binary distribution
    samples.append(Sample(
        prompt=[Message(role="user", content="You must respond with either 'yes' or 'no'. Answer:")],
        target_distribution={"yes": 0.7, "no": 0.3},
        word1="yes",
        word2="no",
        tvd_tolerance=0.1,
        is_given_words=True,
        case_type="binary_choice"
    ))
    
    # Example 2: Three-way distribution
    samples.append(Sample(
        prompt=[Message(role="user", content="Choose one: red, blue, or green. Your choice:")],
        target_distribution={"red": 0.4, "blue": 0.3, "green": 0.3},
        word1="red",
        word2="blue",
        tvd_tolerance=0.1,
        is_given_words=True,
        case_type="three_way_choice"
    ))
    
    # Example 3: Skewed distribution
    samples.append(Sample(
        prompt=[Message(role="user", content="Respond with A, B, C, or D. Answer:")],
        target_distribution={"A": 0.8, "B": 0.1, "C": 0.05, "D": 0.05},
        word1="A",
        word2="B",
        tvd_tolerance=0.1,
        is_given_words=True,
        case_type="skewed_choice"
    ))
    
    return samples


def main():
    """Main function demonstrating LoRA fine-tuning."""
    
    # Configuration
    model_id = "huggingface/qwen2-7b-instruct"  # Example model
    output_dir = "./lora_finetuned_model"
    
    print("=== LoRA Fine-tuning Example ===")
    print(f"Model: {model_id}")
    print(f"Output directory: {output_dir}")
    
    # Create training samples
    samples = create_training_samples()
    print(f"Created {len(samples)} training samples")
    
    # Example 1: Fine-tune with TVD loss, targeting both attention and MLP
    print("\n--- Example 1: TVD Loss, Both Attention and MLP ---")
    
    from provider_wrapper import LoRAConfig
    
    finetuner = LoRAFineTuner(
        model_id=model_id,
        lora_config=LoRAConfig(target="both"),
        custom_loss=TVDLoss()
    )
    
    # Train the model
    trainer = finetuner.train(
        samples=samples,
        output_dir=output_dir,
        num_epochs=2,
        batch_size=2,
        gradient_accumulation_steps=2,
        warmup_steps=50,
        save_steps=100,
        logging_steps=5
    )
    
    print(f"Training completed! Model saved to {output_dir}")
    
    # Example 2: Fine-tune with KL divergence loss, targeting only attention
    print("\n--- Example 2: KL Loss, Attention Only ---")
    
    kl_finetuner = LoRAFineTuner(
        model_id=model_id,
        lora_config=LoRAConfig(target="attention"),
        custom_loss=KLDivergenceLoss()
    )
    
    # Train with different hyperparameters
    kl_trainer = kl_finetuner.train(
        samples=samples,
        output_dir=f"{output_dir}_kl_attention",
        num_epochs=1,
        batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=25,
        save_steps=50,
        logging_steps=10
    )
    
    print(f"KL training completed! Model saved to {output_dir}_kl_attention")
    
    # Example 3: Custom configuration
    print("\n--- Example 3: Custom Configuration ---")
    
    from provider_wrapper import KLDivergenceLoss, LoRAConfig

    # Create custom LoRA config
    custom_config = LoRAConfig(
        r=32,  # Higher rank
        alpha=64,
        dropout=0.2,
        target="mlp",  # Only MLP layers
        learning_rate=5e-5,
        weight_decay=0.02
    )
    
    # Create custom loss function
    custom_loss = KLDivergenceLoss(temperature=0.8)
    
    # Create fine-tuner with custom config
    custom_finetuner = LoRAFineTuner(
        model_id=model_id,
        lora_config=custom_config,
        custom_loss=custom_loss
    )
    
    # Train with custom configuration
    custom_trainer = custom_finetuner.train(
        samples=samples,
        output_dir=f"{output_dir}_custom",
        num_epochs=1,
        batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=25,
        save_steps=50,
        logging_steps=10
    )
    
    print(f"Custom training completed! Model saved to {output_dir}_custom")
    
    print("\n=== All examples completed! ===")
    print("You can now load the fine-tuned models using:")
    print("finetuner.load_finetuned_model(output_dir)")


if __name__ == "__main__":
    main() 