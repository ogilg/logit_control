#!/usr/bin/env bash
set -euo pipefail

MODEL="gpt-oss-20b"
NUM_EXAMPLES=500

python evals/output_control/evaluate_model.py "$MODEL" --num_examples "$NUM_EXAMPLES" --reasoning_effort low > results/run_low.log 2>&1
python evals/output_control/evaluate_model.py "$MODEL" --num_examples "$NUM_EXAMPLES" --reasoning_effort medium > results/run_medium.log 2>&1
python evals/output_control/evaluate_model.py "$MODEL" --num_examples "$NUM_EXAMPLES" --reasoning_effort high > results/run_high.log 2>&1

echo "Done. Logs: results/run_{low,medium,high}.log"