# LoRA Fine-tuning Support

This package now includes support for LoRA (Low-Rank Adaptation) fine-tuning with custom loss functions designed for distribution matching tasks.

## Features

- **Configurable LoRA targets**: Target attention layers, MLP layers, or both
- **Custom loss functions**: TVD (Total Variation Distance) and KL divergence losses
- **Easy-to-use interface**: Simple API for fine-tuning models
- **Memory efficient**: Uses 8-bit quantization and gradient checkpointing
- **Flexible configuration**: Customizable LoRA parameters and training settings
- **Clean architecture**: Uses standard Trainer with simple custom loss integration

## Quick Start

### Basic Usage

```python
from custom_evals import create_finetuner, Sample, Message

# Create training samples
samples = [
    Sample(
        prompt=[Message(role="user", content="You must respond with either 'yes' or 'no'. Answer:")],
        target_distribution={"yes": 0.7, "no": 0.3},
        word1="yes",
        word2="no",
        tvd_tolerance=0.1,
        is_given_words=True
    )
]

# Create fine-tuner
finetuner = create_finetuner(
    model_id="huggingface/qwen2-7b-instruct",
    target="both",  # Target both attention and MLP
    loss_type="tvd"  # Use TVD loss
)

# Train the model
trainer = finetuner.train(
    samples=samples,
    output_dir="./lora_output",
    num_epochs=3,
    batch_size=4
)
```

### Advanced Configuration

```python
from custom_evals import LoRAConfig, TVDLoss

# Custom LoRA configuration
config = LoRAConfig(
    r=32,  # LoRA rank
    alpha=64,  # LoRA alpha
    dropout=0.2,
    target="attention",  # Only attention layers
    learning_rate=5e-5
)

# Custom loss function
loss_fn = TVDLoss(temperature=0.8)

# Create fine-tuner with custom config
finetuner = LoRAFineTuner(
    model_id="huggingface/qwen2-7b-instruct",
    lora_config=config,
    custom_loss=loss_fn
)
```

## Configuration Options

### LoRA Targets

- `"attention"`: Only target attention layers (q_proj, k_proj, v_proj, o_proj)
- `"mlp"`: Only target MLP layers (gate_proj, up_proj, down_proj)
- `"both"`: Target both attention and MLP layers

### Loss Functions

#### TVD Loss (Total Variation Distance)
Penalizes the model for deviating from the target distribution using TVD.

```python
from custom_evals import TVDLoss

loss_fn = TVDLoss(temperature=1.0)
```

#### KL Divergence Loss
Uses KL divergence to measure distribution differences.

```python
from custom_evals import KLDivergenceLoss

loss_fn = KLDivergenceLoss(temperature=1.0)
```

### LoRA Configuration

```python
from custom_evals import LoRAConfig

config = LoRAConfig(
    r=16,                    # LoRA rank (default: 16)
    alpha=32,                # LoRA alpha (default: 32)
    dropout=0.1,             # Dropout rate (default: 0.1)
    target="both",  # Target layers
    learning_rate=1e-4,      # Learning rate
    weight_decay=0.01,       # Weight decay
    bias="none",             # Bias handling
    gradient_checkpointing=True  # Memory optimization
)

### Using Class Methods

The LoRAConfig class provides convenient class methods for common configurations:

```python
from custom_evals import LoRAConfig

# Attention-only configuration
config = LoRAConfig.attention_only(r=16, alpha=32)

# MLP-only configuration  
config = LoRAConfig.mlp_only(r=16, alpha=32)

# Both layers configuration
config = LoRAConfig.both(r=16, alpha=32)

# High-rank configuration (more capacity)
config = LoRAConfig.high_rank(target="attention")

# Low-rank configuration (more efficient)
config = LoRAConfig.low_rank(target="mlp")
```

## Training Samples

Training samples should be instances of `Sample`:

```python
from custom_evals import Sample, Message

sample = Sample(
    prompt=[Message(role="user", content="Your prompt here")],
    target_distribution={"token1": 0.6, "token2": 0.4},
    word1="token1",
    word2="token2",
    tvd_tolerance=0.1,
    is_given_words=True,
    case_type="example"
)
```

## Example Use Cases

### 1. Binary Choice Training
Train a model to respond with specific probabilities for binary choices.

```python
samples = [
    TrainingSample(
        prompt="Answer yes or no:",
        target_distribution={"yes": 0.8, "no": 0.2}
    )
]
```

### 2. Multi-way Choice Training
Train a model for multi-way choices with specific distributions.

```python
samples = [
    TrainingSample(
        prompt="Choose A, B, C, or D:",
        target_distribution={"A": 0.5, "B": 0.3, "C": 0.15, "D": 0.05}
    )
]
```

### 3. Custom Loss Functions
Loss functions are located in `lora/loss_functions.py`. Implement your own custom loss function by subclassing `CustomLossFunction`:

```python
from custom_evals import CustomLossFunction
import torch.nn.functional as F

class MyCustomLoss(CustomLossFunction):
    def _compute_loss_formula(self, target_logits, target_probs):
        """Implement only the specific loss formula."""
        # Convert to probabilities
        probs = F.softmax(target_logits / self.temperature, dim=-1)
        
        # Your custom loss formula here
        custom_loss = torch.sum((probs - target_probs) ** 2)  # Example: MSE
        
        return custom_loss
```

The loss functions automatically handle two cases:
- **Given words**: Uses specific target tokens from the distribution (e.g., {"yes": 0.7, "no": 0.3})
- **Random words**: Finds top 2 tokens by log probability and compares to target distribution (e.g., {"first_word": 0.6, "second_word": 0.4})

## Memory Requirements

The package is optimized for memory efficiency:
- Uses 8-bit quantization by default
- Implements gradient checkpointing
- Supports device mapping for multi-GPU setups

For models up to ~20GB, this should work on a 48GB GPU.

## Loading Fine-tuned Models

After training, you can load the fine-tuned model:

```python
# Load the fine-tuned model
finetuner.load_finetuned_model("./lora_output")

# Use the model for inference
# (The model will now follow the learned distributions)
```

## Dependencies

Make sure you have the following packages installed:
- `torch`
- `transformers`
- `peft`
- `accelerate`

## Example Script

See `example_lora_finetuning.py` for a complete working example that demonstrates:
- Basic LoRA fine-tuning
- Different target configurations
- Custom loss functions
- Advanced training settings 