#!/bin/bash

# Script to evaluate cheap open source models on output_control task
# Both plain and sp variants

echo "Starting output_control evaluation on cheap open source models..."

# List of cheap models to evaluate (base and chat/instruct versions)
MODELS=(
    "mistral-7b"
)

# Convert array to comma-separated string
MODELS_STR=$(IFS=,; echo "${MODELS[*]}")

echo "Models to evaluate:"
for model in "${MODELS[@]}"; do
    echo "  - $model"
done
echo

echo "Running evaluation with both plain and sp variants..."
echo "Command: sad run --tasks output_control --models $MODELS_STR --variants plain"
sad run --tasks output_control --models "$MODELS_STR" --variants "plain"
echo

echo "Command: sad run --tasks output_control --models $MODELS_STR --variants sp"
sad run --tasks output_control --models "$MODELS_STR" --variants "sp"

if [ $? -eq 0 ]; then
    echo
    echo "Evaluation completed successfully!"
    echo
    echo "Showing results..."
    echo "=================="
    sad results --tasks output_control
else
    echo
    echo "Evaluation failed!"
    exit 1
fi 