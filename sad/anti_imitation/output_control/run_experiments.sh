#!/bin/bash

# List of models to run experiments on
models=(
    "qwen2.5-3b"
    "qwen2.5-3b-instruct"
    "qwen2.5-7b"
    "qwen2.5-7b-instruct"
    "llama-3.1-8b"
    "llama-3.1-8b-instruct"
    "llama-3.2-3b"
    "llama-3.2-3b-instruct"
    "deepseek-coder-7b"
    "deepseek-coder-7b-instruct"
    "deepseek-llm-7b"
    "deepseek-llm-7b-instruct"
    "deepseek-llm-7b-chat"
    "deepseek-llm-7b-chat-instruct"
)

for model in "${models[@]}"; do
    # a) Evaluate base model
    python sad/anti_imitation/output_control/run_experiment.py --models "$model" --num_examples 25
    
    # b) Fine-tune with LoRA
    python sad/anti_imitation/output_control/lora_finetune.py --model_name "$model" --num_given_examples 50 --num_random_examples 50 --num_icl_examples 50  --lora_r 16 --lora_target "both" --num_epochs 3
    
    # c) Evaluate fine-tuned model
    lora_dir="./lora_output/lora_output_${model//[.-]/_}"
    python sad/anti_imitation/output_control/run_experiment.py --models "$model" --num_examples 25 --lora_adapter "$lora_dir"
    
    # d) Kill all Python processes
    pkill -f python
    rm -rf ~/.cache/huggingface
done
