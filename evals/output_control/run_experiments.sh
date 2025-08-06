#!/bin/bash

# Set environment variables for stable downloads
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_DOWNLOAD_MAX_WORKERS=4
export TRANSFORMERS_VERBOSITY=warning
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DISABLE_TELEMETRY=1

# Function to run command with timeout and retry
run_with_timeout_and_retry() {
    local cmd="$1"
    local max_attempts=4
    local timeout_seconds=1500  # 25 minutes
    
    for attempt in $(seq 1 $max_attempts); do
        echo "Attempt $attempt/$max_attempts: $cmd"
        
        # Run command with timeout
        if timeout $timeout_seconds bash -c "$cmd"; then
            echo "Command succeeded on attempt $attempt"
            return 0
        else
            echo "Command failed on attempt $attempt (timeout or error)"
            
            # Clear cache and kill processes between attempts
            echo "Clearing cache and killing processes..."
            pkill -f python 2>/dev/null || true
            rm -rf ~/.cache/huggingface 2>/dev/null || true
            sleep 10
            
            if [ $attempt -eq $max_attempts ]; then
                echo "All attempts failed for: $cmd"
                return 1
            fi
        fi
    done
}

# Function to preload model with retry
preload_model() {
    local model="$1"
    
    local cmd="python evals/output_control/preload_models.py --models \"$model\""
    
    if run_with_timeout_and_retry "$cmd"; then
        echo "Model preloading completed successfully for $model"
        return 0
    else
        echo "Model preloading failed for $model, skipping to next model"
        return 1
    fi
}

# Function to evaluate model with retry
evaluate_model() {
    local model="$1"
    local lora_adapter="$2" 
    local num_examples="$3"
    
    local cmd="python evals/output_control/run_experiment.py --models \"$model\" --num_examples $num_examples"
    
    if [ -n "$lora_adapter" ]; then
        cmd="$cmd --lora_adapter \"$lora_adapter\""
    fi
    
    if run_with_timeout_and_retry "$cmd"; then
        echo "Evaluation completed successfully for $model"
        return 0
    else
        echo "Evaluation failed for $model, skipping to next model"
        return 1
    fi
}

# Function to fine-tune model with retry
fine_tune_model() {
    local model="$1"
    
    local cmd="python sad/anti_imitation/output_control/lora_finetune.py --model_name \"$model\" --num_given_examples 50 --num_random_examples 50 --num_icl_examples 50 --lora_r 16 --lora_target \"both\" --num_epochs 3"
    
    if run_with_timeout_and_retry "$cmd"; then
        echo "Fine-tuning completed successfully for $model"
        return 0
    else
        echo "Fine-tuning failed for $model, skipping to next model"
        return 1
    fi
}

# List of models to run experiments on
models=(
    "llama-3.2-3b"
    "llama-3.2-3b-instruct"
    "llama-3.1-8b-instruct"
    "llama-3.1-8b"
    "deepseek-coder-7b"
    "deepseek-coder-7b-instruct"
    "deepseek-llm-7b"
    "deepseek-llm-7b-instruct"
    "deepseek-llm-7b-chat"
    "deepseek-llm-7b-chat-instruct"
)

echo "Starting experiments with robust download handling"
echo "Environment variables:"
echo "  HF_HUB_DOWNLOAD_TIMEOUT=$HF_HUB_DOWNLOAD_TIMEOUT"
echo "  HF_HUB_DOWNLOAD_MAX_WORKERS=$HF_HUB_DOWNLOAD_MAX_WORKERS"
echo "  TRANSFORMERS_VERBOSITY=$TRANSFORMERS_VERBOSITY"
echo ""

for model in "${models[@]}"; do
    echo "=========================================="
    echo "Processing model: $model"
    echo "=========================================="
    
    # Clear any lingering processes and cache before starting
    pkill -f python 2>/dev/null || true
    rm -rf ~/.cache/huggingface 2>/dev/null || true
    sleep 5
    
    # 0) Preload model first
    echo "Step 0: Preloading model $model"
    if ! preload_model "$model"; then
        echo "Model preloading failed for $model, skipping to next model"
        continue
    fi
    
    # a) Evaluate base model  
    echo "Step 1: Evaluating base model $model"
    if ! evaluate_model "$model" "" 25; then
        echo "Base model evaluation failed for $model, skipping to next model"
        continue
    fi
    
    # b) Fine-tune with LoRA
    echo "Step 2: Fine-tuning model $model with LoRA"
    if ! fine_tune_model "$model"; then
        echo "Fine-tuning failed for $model, skipping to next model"
        continue
    fi
    
    # c) Evaluate fine-tuned model
    echo "Step 3: Evaluating fine-tuned model $model"
    lora_dir="./lora_output/lora_output_${model//[.-]/_}"
    if ! evaluate_model "$model" "$lora_dir" 25; then
        echo "Fine-tuned model evaluation failed for $model"
    fi
    
    # d) Clean up after each model
    echo "Cleaning up after $model"
    pkill -f python 2>/dev/null || true
    rm -rf ~/.cache/huggingface 2>/dev/null || true
    
done

echo "All experiments completed!"
