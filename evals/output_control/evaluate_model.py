#!/usr/bin/env python3
"""
CLI to evaluate a single model (base or LoRA fine-tuned) using the output-control experiment.

- Pass a base model ID (e.g., "llama-3.1-8b")
- Optionally pass a LoRA adapter path or Hub repo (e.g., "username/model-lora")
"""

import argparse
import json
import os
import sys

# Ensure we can import sibling module when running from repo root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from run_eval import (
    run_experiment_for_model,
    update_results_csv,
    update_simple_metrics_csv,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a single model with the output-control experiment")
    parser.add_argument("model", help="Model ID to evaluate (e.g., llama-3.1-8b)")
    parser.add_argument("--num_examples", type=int, default=10, help="Examples per experiment type (default: 10)")
    parser.add_argument(
        "--lora_adapter",
        type=str,
        default=None,
        help="Path or HF repo ID of a LoRA adapter to load during evaluation",
    )
    parser.add_argument(
        "--save_individual",
        action="store_true",
        help="Save individual model results to JSON (in addition to CSV updates)",
    )

    args = parser.parse_args()

    # Run the experiment for a single model
    results = run_experiment_for_model(
        model_id=args.model,
        num_examples=args.num_examples,
        lora_adapter_path=args.lora_adapter,
    )

    # Update CSVs (same behavior as the multi-model script)
    update_results_csv(args.model, results)
    update_simple_metrics_csv(args.model, results, args.lora_adapter)

    # Optionally save individual JSON results
    if args.save_individual:
        output_file = f"evals/output_control_results_{args.model.replace('/', '_')}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Individual model results saved to {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())


