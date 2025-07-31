"""
LoRA configuration management for fine-tuning.

This module provides configuration classes and utilities for setting up LoRA fine-tuning
with customizable target layers (attention, MLP, or both).
"""

from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field, computed_field


class LoRAConfig(BaseModel):
    """Configuration for LoRA fine-tuning with Pydantic validation."""
    
    r: int = Field(
        default=16,
        description="LoRA rank parameter - controls the rank of the low-rank adaptation"
    )
    
    alpha: int = Field(
        default=32,
        description="LoRA alpha parameter - scaling factor for the LoRA weights"
    )
    
    dropout: float = Field(
        default=0.1,
        description="Dropout rate for LoRA layers to prevent overfitting"
    )
    
    target: Literal["attention", "mlp", "both"] = Field(
        default="both",
        description="Target layers for LoRA adaptation: 'attention' for attention layers only, 'mlp' for MLP layers only, 'both' for both"
    )
    
    learning_rate: float = Field(
        default=1e-4,
        description="Learning rate for LoRA training"
    )
    
    weight_decay: float = Field(
        default=0.01,
        description="Weight decay for LoRA training to prevent overfitting"
    )
    
    bias: Literal["none", "all", "lora_only"] = Field(
        default="none",
        description="Bias handling in LoRA layers: 'none' for no bias, 'all' for all biases, 'lora_only' for LoRA biases only"
    )
    
    task_type: str = Field(
        default="CAUSAL_LM",
        description="Task type for the model (typically CAUSAL_LM for language models)"
    )
    
    gradient_checkpointing: bool = Field(
        default=True,
        description="Whether to use gradient checkpointing for memory efficiency"
    )
    
    use_flash_attention: bool = Field(
        default=False,
        description="Whether to use flash attention if available (requires flash-attn package)"
    )
    
    @computed_field
    @property
    def target_modules(self) -> List[str]:
        """Get target modules based on the target configuration."""
        if self.target == "attention":
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif self.target == "mlp":
            return ["gate_proj", "up_proj", "down_proj"]
        elif self.target == "both":
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            raise ValueError(f"Unknown target: {self.target}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for PEFT."""
        return {
            "r": self.r,
            "lora_alpha": self.alpha,
            "lora_dropout": self.dropout,
            "bias": self.bias,
            "task_type": self.task_type,
            "target_modules": self.target_modules,
        }
    
    def get_training_args(self) -> Dict[str, Any]:
        """Get training arguments for the fine-tuning process."""
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_checkpointing": self.gradient_checkpointing,
        }
