#!/bin/bash

# Script to run output_control task on specified models with both variants
# Usage: ./run_output_control.sh [model1,model2,...]
# Example: ./run_output_control.sh phi-2
# Example: ./run_output_control.sh phi-2,deepseek-coder-7b,qwen2-7b

# Default models if none provided
DEFAULT_MODELS="phi-2"

# Function to clear memory (platform-specific)
clear_memory() {
    echo "Clearing memory..."
    
    # Clear GPU memory and Python objects
    python3 -c "
import gc
import torch
import sys

# Force garbage collection
gc.collect()

# Clear GPU memory if available
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f'GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0):,} bytes')
else:
    print('No GPU available')

# Clear Python objects
print(f'Python objects: {len(gc.get_objects())}')
" 2>/dev/null || true
    
    # For Linux - clear page cache, dentries and inodes (requires sudo)
    if command -v sudo >/dev/null 2>&1; then
        echo "Clearing page cache..."
        sudo sh -c "echo 3 > /proc/sys/vm/drop_caches" 2>/dev/null || true
    fi
    
    # For Linux
    if command -v sync >/dev/null 2>&1; then
        sync
    fi
    
    echo "Memory cleared."
}

# Function to run a single model with both variants
run_model() {
    local model=$1
    echo "=========================================="
    echo "Running output_control task for model: $model"
    echo "=========================================="
    
    # Run plain variant
    echo "Running plain variant..."
    timeout 3600 sad run --tasks output_control --models "$model" --variants plain
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to run plain variant for $model"
        return 1
    fi
    
    # Run sp variant (same model, no memory clearing needed)
    echo "Running sp variant..."
    timeout 3600 sad run --tasks output_control --models "$model" --variants sp
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to run sp variant for $model"
        return 1
    fi
    
    echo "Completed both variants for $model"
    echo "=========================================="
    echo ""
}

# Main script
main() {
    # Get models from command line or use default
    MODELS=${1:-$DEFAULT_MODELS}
    
    # Convert comma-separated string to array
    IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"
    
    echo "Starting output_control task evaluation"
    echo "Models to run: ${MODEL_ARRAY[*]}"
    echo "Total models: ${#MODEL_ARRAY[@]}"
    echo ""
    
    # Run each model
    for model in "${MODEL_ARRAY[@]}"; do
        # Trim whitespace
        model=$(echo "$model" | xargs)
        
        if [ -z "$model" ]; then
            continue
        fi
        
        echo "Processing model: $model"
        
        # Run the model with both variants
        run_model "$model"
        
        # Clear memory after each model (except the last one)
        if [ "$model" != "${MODEL_ARRAY[-1]}" ]; then
            clear_memory
        fi
    done
    
    echo "=========================================="
    echo "All models completed!"
    echo "You can now run 'sad results --tasks output_control' to see the results"
    echo "=========================================="
}

# Run the main function with all arguments
main "$@" 