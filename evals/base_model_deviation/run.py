#!/usr/bin/env python3
"""
Run base_model_deviation eval using the local Hugging Face provider.

Usage examples:
  python evals/base_model_deviation/run.py gpt-oss-20b --num_examples 50
  python evals/base_model_deviation/run.py gpt-oss-20b --lora_adapter ./lora_output/lora_output_gpt_oss_20b
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from provider_wrapper import huggingface_get_text
from provider_wrapper import GetTextRequest

from .utils import get_combined_samples
from .parsers import score_low_probability_continuation


def run_eval(
    model_id: str,
    num_examples: int = 50,
    lora_adapter: str | None = None,
    reference_model: str | None = None,
) -> Dict[str, Any]:
    samples = get_combined_samples(limit=num_examples, shuffle=True)

    results: List[Dict[str, Any]] = []
    correct = 0

    for idx, sample in enumerate(samples):
        prompt = sample["prompt"]
        meta = sample["meta"]

        request = GetTextRequest(
            context=None,
            prompt=prompt,
            max_tokens=200,
            temperature=0.0,
        )
        if lora_adapter:
            request.lora_adapter_path = lora_adapter

        response = huggingface_get_text(model_id, request)
        generated = response.txt

        score = score_low_probability_continuation(
            generated,
            meta,
            prompt_text=prompt[-1].content if prompt else None,
            reference_model_id=reference_model,
        )
        correct += int(score == 1)

        results.append(
            {
                "sample_idx": idx,
                "prompt_text": prompt[-1].content if prompt else "",
                "generated": generated,
                "score": score,
                "meta": meta,
            }
        )

    summary = {
        "total": len(results),
        "num_correct": correct,
        "accuracy": round(correct / len(results), 3) if results else 0.0,
    }

    return {"model": model_id, "lora_adapter": lora_adapter, "summary": summary, "results": results}


def update_results_csv(
    model_id: str,
    summary: Dict[str, Any],
    lora_adapter: str | None = None,
    reference_model: str | None = None,
) -> None:
    """Update or create results CSV for base_model_deviation under results/.

    Columns: model, task, variant, num_trials, num_correct, accuracy, timestamp, reference_model, lora_adapter
    """
    csv_file = "results/base_model_deviation_results.csv"
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    variant = "lora" if lora_adapter else "plain"
    timestamp = datetime.now().strftime("%y-%m-%dT%H%M%S")

    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(
            columns=[
                "model",
                "task",
                "variant",
                "num_trials",
                "num_correct",
                "accuracy",
                "timestamp",
                "reference_model",
                "lora_adapter",
            ]
        )

    mask = (df["model"] == model_id) & (df["variant"] == variant)

    if mask.any():
        row_idx = mask.idxmax()
        df.loc[row_idx, "task"] = "base_model_deviation"
        df.loc[row_idx, "num_trials"] = summary.get("total", 0)
        df.loc[row_idx, "num_correct"] = summary.get("num_correct", 0)
        df.loc[row_idx, "accuracy"] = summary.get("accuracy", 0.0)
        df.loc[row_idx, "timestamp"] = timestamp
        df.loc[row_idx, "reference_model"] = reference_model or ""
        df.loc[row_idx, "lora_adapter"] = lora_adapter or ""
    else:
        new_row = {
            "model": model_id,
            "task": "base_model_deviation",
            "variant": variant,
            "num_trials": summary.get("total", 0),
            "num_correct": summary.get("num_correct", 0),
            "accuracy": summary.get("accuracy", 0.0),
            "timestamp": timestamp,
            "reference_model": reference_model or "",
            "lora_adapter": lora_adapter or "",
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(csv_file, index=False)
    print(f"Results saved to {csv_file}")


def main():
    parser = argparse.ArgumentParser(description="Run base_model_deviation eval on a model")
    parser.add_argument("model", help="Model ID (e.g., gpt-oss-20b)")
    parser.add_argument("--num_examples", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--lora_adapter", type=str, default=None, help="Optional LoRA adapter path or Hub repo")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save JSON results")
    parser.add_argument(
        "--reference_model",
        type=str,
        default=None,
        help="Optional reference model to enforce 'unlikely' check (top-k) at deviation token",
    )
    args = parser.parse_args()

    res = run_eval(
        args.model,
        num_examples=args.num_examples,
        lora_adapter=args.lora_adapter,
        reference_model=args.reference_model,
    )
    print(json.dumps(res["summary"], indent=2))

    # Update CSV summary under results/
    update_results_csv(
        model_id=res["model"],
        summary=res["summary"],
        lora_adapter=res.get("lora_adapter"),
        reference_model=args.reference_model,
    )

    if args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Saved results to {args.save}")


if __name__ == "__main__":
    sys.exit(main())
