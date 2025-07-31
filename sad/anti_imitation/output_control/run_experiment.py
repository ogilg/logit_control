#!/usr/bin/env python3
"""
Simple CLI script to run the output control experiment on multiple models.
"""

import argparse
import os
import sys
import json
import csv
import pandas as pd
from datetime import datetime
from utils import get_combined_samples, get_model_probabilities, sample_model_text
from custom_evals import (
    HUGGINGFACE_MODEL_MAPPING, 
    huggingface_get_probs,
    huggingface_get_text,
    GetProbsRequest,
    GetTextRequest,
    Message
)
from data_models import Sample
from parsers import tvd_parser
from tqdm import tqdm
import math
import torch
import gc
import shutil
from typing import List, Dict, Any


def round_floats(obj):
    """Recursively round floats to 3 decimal places in nested dictionaries and lists."""
    if isinstance(obj, float):
        return round(obj, 3)
    elif isinstance(obj, dict):
        return {key: round_floats(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(item) for item in obj]
    else:
        return obj

def calculate_standard_error(p, n):
    """Calculate standard error using Bernoulli formula: sqrt(p * (1-p) / n)"""
    if n == 0 or p <= 0 or p >= 1:
        return 0.0
    return math.sqrt(p * (1 - p) / n)

def clear_gpu_memory():
    """Clear GPU memory after each model."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("GPU memory cleared")

def clear_huggingface_cache():
    """Clear HuggingFace cache to free disk space."""
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    if os.path.exists(cache_dir):
        try:
            # Only clear the hub cache, keep tokenizers cache for efficiency
            hub_cache = os.path.join(cache_dir, "hub")
            if os.path.exists(hub_cache):
                shutil.rmtree(hub_cache)
                print("HuggingFace hub cache cleared")
        except Exception as e:
            print(f"Warning: Could not clear HuggingFace cache: {e}")

def update_results_csv(model_id, results):
    """Update the main results CSV file (original format without TVD stats)."""
    csv_file = "evals/output_control_results.csv"
    
    # Calculate metrics
    summary = results['summary']
    total_experiments = summary['total_experiments']
    within_tolerance = summary['within_tolerance']
    
    # Count responses (no invalid responses since we're using token probabilities)
    correct_count = 0
    incorrect_count = 0
    
    for result in results['results']:
        if result['within_tolerance']:
            correct_count += 1
        else:
            incorrect_count += 1
    
    # Calculate old-style scores (invalid_count is always 0 for token probabilities)
    invalid_count = 0
    score = within_tolerance / total_experiments if total_experiments > 0 else 0.0
    score_n = correct_count / (correct_count + incorrect_count) if (correct_count + incorrect_count) > 0 else 0.0
    score_p = correct_count / (correct_count + incorrect_count) if (correct_count + incorrect_count) > 0 else 0.0
    
    # Calculate standard errors for old scores
    std = calculate_standard_error(score, total_experiments)
    std_n = calculate_standard_error(score_n, total_experiments)
    std_p = calculate_standard_error(score_p, total_experiments)
    
    # Round scores to 3 decimal places
    score = round(score, 3)
    score_n = round(score_n, 3)
    score_p = round(score_p, 3)
    std = round(std, 3)
    std_n = round(std_n, 3)
    std_p = round(std_p, 3)
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%y-%m-%dT%H%M%S")
    
    # Read existing CSV
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        # Create new DataFrame with original columns (no TVD stats)
        df = pd.DataFrame(columns=[
            'model', 'task', 'variant', 'score', 'score_n', 'score_p', 
            'num_trials', 'std', 'std_n', 'std_p', 'correct', 'incorrect', 
            'invalid', 'timestamp', 'human_baseline', 'human_answer_baseline'
        ])
    
    # Find row to update (match model only, no variant)
    mask = (df['model'] == model_id)
    
    if mask.any():
        # Update existing row
        row_idx = mask.idxmax()
        df.loc[row_idx, 'variant'] = 'plain'
        df.loc[row_idx, 'score'] = score
        df.loc[row_idx, 'score_n'] = score_n
        df.loc[row_idx, 'score_p'] = score_p
        df.loc[row_idx, 'num_trials'] = total_experiments
        df.loc[row_idx, 'std'] = std
        df.loc[row_idx, 'std_n'] = std_n
        df.loc[row_idx, 'std_p'] = std_p
        df.loc[row_idx, 'correct'] = correct_count
        df.loc[row_idx, 'incorrect'] = incorrect_count
        df.loc[row_idx, 'invalid'] = invalid_count
        df.loc[row_idx, 'timestamp'] = timestamp
    else:
        # Create new row
        new_row = {
            'model': model_id,
            'task': 'output_control',
            'variant': 'plain',
            'score': score,
            'score_n': score_n,
            'score_p': score_p,
            'num_trials': total_experiments,
            'std': std,
            'std_n': std_n,
            'std_p': std_p,
            'correct': correct_count,
            'incorrect': incorrect_count,
            'invalid': invalid_count,
            'timestamp': timestamp,
            'human_baseline': '',
            'human_answer_baseline': ''
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save updated CSV
    df.to_csv(csv_file, index=False)
    print(f"Results saved to {csv_file}")

def update_simple_metrics_csv(model_id, results):
    """Update the simple metrics CSV file with basic counts and TVD stats, separated by case type."""
    csv_file = "evals/output_control_simple_metrics.csv"
    
    # Separate results by case type
    given_results = [r for r in results['results'] if r.get('case_type') == 'given_words']
    random_results = [r for r in results['results'] if r.get('case_type') == 'random_words']
    icl_random_results = [r for r in results['results'] if r.get('case_type') == 'icl_random_words']
    
    # Function to calculate metrics for a subset of results
    def calculate_case_metrics(case_results, case_type):
        if not case_results:
            return None
        
        # Calculate TVD statistics
        tvds = [r['tvd'] for r in case_results]
        avg_tvd = sum(tvds) / len(tvds) if tvds else 0.0
        std_tvd = pd.Series(tvds).std() if len(tvds) > 1 else 0.0
        
        # Count responses
        correct_count = sum(1 for r in case_results if r['within_tolerance'])
        incorrect_count = len(case_results) - correct_count
        
        # Round to 3 decimal places
        avg_tvd = round(avg_tvd, 3)
        std_tvd = round(std_tvd, 3)
        
        return {
            'model': model_id,
            'variant': 'plain',
            'case_type': case_type,
            'num_correct': correct_count,
            'num_incorrect': incorrect_count,
            'num_trials': len(case_results),
            'avg_tvd': avg_tvd,
            'std_tvd': std_tvd
        }
    
    # Calculate metrics for all three cases
    given_metrics = calculate_case_metrics(given_results, 'given_words')
    random_metrics = calculate_case_metrics(random_results, 'random_words')
    icl_random_metrics = calculate_case_metrics(icl_random_results, 'icl_random_words')
    
    # Read existing CSV
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        # Create new DataFrame with case_type column
        df = pd.DataFrame(columns=[
            'model', 'variant', 'case_type', 'num_correct', 'num_incorrect', 
            'num_trials', 'avg_tvd', 'std_tvd'
        ])
    
    # Update or add rows for all three cases
    for metrics in [given_metrics, random_metrics, icl_random_metrics]:
        if metrics is None:
            continue
            
        # Find existing row for this model and case type
        mask = (df['model'] == model_id) & (df['case_type'] == metrics['case_type'])
        
        if mask.any():
            # Update existing row
            row_idx = mask.idxmax()
            for key, value in metrics.items():
                if key != 'model' and key != 'case_type':  # Don't update these keys
                    df.loc[row_idx, key] = value
        else:
            # Create new row
            df = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)
    
    # Save updated CSV
    df.to_csv(csv_file, index=False)
    print(f"Simple metrics saved to {csv_file}")


def run_experiment(model_id: str, samples: List[Sample], num_examples: int = 10) -> Dict[str, Any]:
    """Run the output control experiment on a model using proper TVD calculation."""
    
    # Extract model name from model_id
    if model_id.startswith("huggingface/"):
        model_name = model_id.split("/")[1]
    else:
        model_name = model_id
    
    if model_name not in HUGGINGFACE_MODEL_MAPPING:
        raise ValueError(f"Model {model_id} not supported by HuggingFace provider")
    
    results = []
    
    for sample_idx, sample in enumerate(tqdm(samples, desc="Running experiments")):
        # Get model probabilities for the prompt
        response_probs = get_model_probabilities(model_name, sample.prompt, temperature=1.0)
        
        # Sample text to see what the model is actually outputting
        sample_text = sample_model_text(model_name, sample.prompt, max_tokens=10, temperature=1.0)
        
        # Prepare sample data for the TVD parser
        sample_data = {
            'prompt_content': sample.prompt[-2].content,  # Get content from the Message object
            'target_distribution': sample.target_distribution,
            'tvd_tolerance': sample.tvd_tolerance,
            'is_given_words': sample.is_given_words,
            'word1': sample.word1,
            'word2': sample.word2,
            'seed': sample.seed
        }

        # Get parser result
        parser_result = tvd_parser(response_probs, sample_data)
        
        # Create result dict
        result = {
            'sample_idx': sample_idx,
            'is_given_words': sample.is_given_words,
            'word1': sample.word1,
            'word2': sample.word2,
            'target_distribution': sample.target_distribution,
            'actual_distribution': dict(zip(parser_result.get('top_tokens', []), parser_result.get('top_probs', []))),
            'tvd_tolerance': sample.tvd_tolerance,
            'tvd': parser_result.get('tvd_diff', float('inf')),
            'within_tolerance': parser_result['score'] == 1,
            'responses': parser_result.get('top_tokens', []),
            'sampled_text': sample_text,  # What the model actually outputs
            'seed': sample.seed,
            'case_type': sample.case_type
        }
        
        results.append(result)
    
    # Calculate summary
    total_experiments = len(results)
    within_tolerance = sum(1 for r in results if r['within_tolerance'])
    avg_tvd = sum(r['tvd'] for r in results) / total_experiments if total_experiments > 0 else 0
    
    return {
        'model_id': model_id,
        'num_samples': num_examples,
        'temperature': 1.0,
        'results': results,
        'summary': {
            'total_experiments': total_experiments,
            'within_tolerance': within_tolerance,
            'avg_tvd': avg_tvd
        }
    }


def run_experiment_for_model(model_id, num_examples):
    """Run experiment for a single model and return results."""
    print(f"Running output control experiment on {model_id}")
    print(f"Parameters: num_examples={num_examples}")
    
    # Get samples and run experiment
    samples = get_combined_samples(num_examples=num_examples)
    results = run_experiment(model_id, samples, num_examples)
    
    # Print summary
    summary = results['summary']
    print(f"Experiment Summary:")
    print(f"Total experiments: {summary['total_experiments']}")
    print(f"Within TVD tolerance: {summary['within_tolerance']} ({summary['within_tolerance']/summary['total_experiments']*100:.1f}%)")
    print(f"Average TVD: {summary['avg_tvd']:.3f}")
    
    # Count experiment types
    given_words = sum(1 for r in results['results'] if r.get('case_type') == 'given_words')
    random_words = sum(1 for r in results['results'] if r.get('case_type') == 'random_words')
    icl_random_words = sum(1 for r in results['results'] if r.get('case_type') == 'icl_random_words')
    print(f"Given words experiments: {given_words}")
    print(f"Random words experiments: {random_words}")
    print(f"ICL random words experiments: {icl_random_words}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run output control experiment on multiple models")
    parser.add_argument("--models", nargs="+", 
                       help="List of model IDs (e.g., phi-2 qwen2-7b). If not provided, uses default model list.")

    parser.add_argument("--num_examples", type=int, default=10, 
                       help="Number of examples per experiment type (default: 10)")
    parser.add_argument("--save_individual", action="store_true",
                       help="Save individual model results to JSON files")
    args = parser.parse_args()

    # Default model list - pairs of base and instruction-tuned models under 30GB
    default_models = [
        "deepseek-llm-7b-chat",
        "deepseek-coder-7b",
        "deepseek-coder-7b-instruct",
        "qwen2-7b",
        "qwen2-7b-instruct",
        "qwen2-3b",
        "qwen2-3b-instruct",
        "llama-3.1-8b",
        "llama-3.1-8b-instruct",
        "llama-3.2-3b",
        "llama-3.2-3b-instruct",
    ]

    # Use provided models or default list
    models_to_run = args.models if args.models else default_models

    print(f"Running experiments on {len(models_to_run)} models: {', '.join(models_to_run)}")
    print(f"Examples per experiment type: {args.num_examples}")
    print("-" * 60)

    all_results = {}
    
    for model_id in models_to_run:
        print(f"\n{'='*60}")
        print(f"Running model: {model_id}")
        print(f"{'='*60}")
        
        try:
            results = run_experiment_for_model(model_id, args.num_examples)
            all_results[model_id] = results
            
            # Update results CSV files
            update_results_csv(model_id, results)
            update_simple_metrics_csv(model_id, results)
            
            # Save individual model results (always save, not just when requested)
            model_output_file = f"evals/output_control_results_{model_id.replace('/', '_')}.json"
            with open(model_output_file, 'w') as f:
                json.dump(round_floats(results), f, indent=2)
            print(f"Individual model results saved to {model_output_file}")
            
            # Clear GPU memory and HuggingFace cache after model is done
            clear_gpu_memory()
            clear_huggingface_cache()
            print(f"Completed {model_id}")
            print("-" * 60)
            
        except Exception as e:
            print(f"Error running experiment for {model_id}: {e}")
            print("-" * 60)
            continue

    # Save combined results
    combined_file = f"evals/output_control_combined_results.json"
    with open(combined_file, 'w') as f:
        json.dump(round_floats(all_results), f, indent=2)
    print(f"Combined results saved to {combined_file}")

    print("\nAll experiments completed!")
    print(f"Results saved to evals/output_control_results.csv")
    print(f"Simple metrics saved to evals/output_control_simple_metrics.csv")

if __name__ == "__main__":
    sys.exit(main()) 