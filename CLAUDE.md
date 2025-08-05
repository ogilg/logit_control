# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project studying "anti-imitation" in Large Language Models - the capacity for models to control their output distributions rather than following standard next-token prediction patterns. The core research question is whether models can learn to control their logits when fine-tuned, and if this generalizes to other sophisticated reasoning tasks.

## Architecture

The project consists of two main Python packages:

### 1. Main Project (`sadcode` package)
- Uses setuptools build system with `pyproject.toml`
- Main entry point: `sad.main:main` CLI command
- Dependencies include PyTorch, Transformers, HuggingFace ecosystem, and evaluation tools

### 2. Provider Wrapper Package (`provider_wrapper/`)
- Separate package providing HuggingFace model evaluation functionality
- Core modules:
  - `huggingface_provider.py`: Local GPU model loading with 4-bit quantization, LoRA adapter support, and both text generation and probability extraction
  - `lora_finetuner.py`: LoRA fine-tuning with custom loss functions (TVD loss for logit control)
  - `loss_functions.py`: Custom loss implementations for training models to control output distributions
  - `data_models.py`: Pydantic models and configuration classes

### 3. Evaluation Framework (`evals/`)
- `output_control/`: Main experiments for logit control evaluation
  - `run_experiment.py`: CLI for running multi-model experiments with TVD (Total Variation Distance) metrics
  - `lora_finetune.py`: Fine-tuning scripts for training models on logit control tasks
  - `utils.py`: Probability extraction and sampling utilities
- `base_model_deviation/`: Additional evaluation experiments

## Key Concepts

- **Output Control Task**: Models are given prompts asking them to control their next-token probability distribution to match specific targets
- **TVD (Total Variation Distance)**: Primary metric for measuring how well models control their output distributions
- **LoRA Fine-tuning**: Used to train models on logit control with custom loss functions that penalize divergence from target distributions
- **Three Case Types**:
  - `given_words`: Models choose between two explicitly provided words
  - `random_words`: Models control distribution over top-2 tokens without explicit words
  - `icl_random_words`: In-context learning variant of random words

## Development Commands

### Installation
```bash
# Install main package in development mode
pip install -e .

# Install provider wrapper package
pip install -e provider_wrapper/
```

### Running Experiments
```bash
# Run output control experiments on default model set
python evals/output_control/run_experiment.py

# Run on specific models with custom sample count
python evals/output_control/run_experiment.py --models llama-3.1-8b qwen2-7b --num_examples 20

# Run with LoRA adapter
python evals/output_control/run_experiment.py --lora_adapter ./lora_output_model_name

# Fine-tune a model with LoRA
python evals/output_control/lora_finetune.py --model llama-3.1-8b --num_samples 500
```

### Results and Output
- Individual model results: `results/output_control_results_{model}.json`
- Combined results: `results/output_control_combined_results.json`
- CSV summaries: `results/output_control_results.csv`, `results/output_control_simple_metrics.csv`

## Model Support

The project primarily works with open-source models in the 3-8B parameter range for compute efficiency:
- Llama 3.1/3.2 (3B, 8B) - base and instruct variants
- Qwen 2/2.5 (3B, 7B) - base and instruct variants
- DeepSeek models (7B) - base, chat, and coder variants

All models use 4-bit quantization via BitsAndBytesConfig for memory efficiency and support LoRA adapters for fine-tuning.

## Important Implementation Details

- Models are loaded with `device_map="auto"` for automatic GPU placement
- Custom loss functions operate on last-token logits for probability control
- TVD tolerance typically set to 0.1 for evaluation
- Results include both accuracy metrics and detailed TVD statistics
- GPU memory is actively managed with cache clearing between model evaluations