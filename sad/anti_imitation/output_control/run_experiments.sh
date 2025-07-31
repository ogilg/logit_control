#!/bin/bash

# List of models to run experiments on
models=(
    "deepseek-llm-7b"
    "deepseek-llm-7b-chat"
    "deepseek-coder-7b"
    "deepseek-coder-7b-instruct"
    "qwen2-7b"
    "qwen2-7b-instruct"
    "qwen2-3b"
    "qwen2-3b-instruct"
    "llama-3.1-8b"
    "llama-3.1-8b-instruct"
    "llama-3.2-3b"
    "llama-3.2-3b-instruct"
)


for model in "${models[@]}"; do
    python sad/anti_imitation/output_control/run_experiment.py --models "$model"
done
