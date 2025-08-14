#!/usr/bin/env bash
set -euo pipefail

MODEL="gpt-oss-20b"
NUM_EXAMPLES=500

pkill -9 python
rm -rf ~/.cache/huggingface/  
python evals/output_control/evaluate_model.py "$MODEL" --num_examples "$NUM_EXAMPLES" --reasoning_effort low
pkill -9 python
rm -rf ~/.cache/huggingface/    
python evals/output_control/evaluate_model.py "$MODEL" --num_examples "$NUM_EXAMPLES" --reasoning_effort medium
pkill -9 python
rm -rf ~/.cache/huggingface/
python evals/output_control/evaluate_model.py "$MODEL" --num_examples "$NUM_EXAMPLES" --reasoning_effort high