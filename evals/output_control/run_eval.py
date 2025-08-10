#!/usr/bin/env python3
"""
Simple CLI script to run the output control experiment on multiple models.
"""

import argparse
import csv
import gc
import json
import math
import os
import shutil
import sys
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import torch
from provider_wrapper.huggingface_provider import extract_model_name
from provider_wrapper import (HUGGINGFACE_MODEL_MAPPING, Sample, validate_model_availability, clear_gpu_memory, clear_huggingface_disk_cache)
from parsers import tvd_parser
from tqdm import tqdm
from utils import (get_combined_samples, get_model_probabilities,
                   sample_model_text)


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



def get_lora_output_dir(model_name: str) -> str:
    """Get the LoRA output directory for a model."""
    clean_name = model_name.replace("/", "_").replace("-", "_").replace(".", "_")
    # Keep consistent with lora_finetune.get_model_output_dir
    return f"./lora_output/lora_output_{clean_name}"

def load_lora_adapter(model_name: str):
    """Load LoRA adapter for a model if it exists."""
    lora_dir = get_lora_output_dir(model_name)
    if os.path.exists(lora_dir):
        print(f"Loading LoRA adapter from {lora_dir}")
        return lora_dir
    else:
        print(f"No LoRA adapter found at {lora_dir}")
        return None

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

def update_simple_metrics_csv(model_id, results, lora_adapter_path=None):
    """Update the simple metrics CSV file with basic counts and TVD stats, separated by case type."""
    csv_file = "evals/output_control_simple_metrics.csv"
    
    # Determine if this is a fine-tuned model
    is_fine_tuned = lora_adapter_path is not None
    
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
            'fine_tuned': is_fine_tuned,
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
        # Create new DataFrame with case_type and fine_tuned columns
        df = pd.DataFrame(columns=[
            'model', 'variant', 'case_type', 'fine_tuned', 'num_correct', 'num_incorrect', 
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


def run_experiment(model_id: str, samples: List[Sample], num_examples: int = 10, lora_adapter_path: str = None) -> Dict[str, Any]:
    """Run the output control experiment on a model using proper TVD calculation."""
    
    # Extract model name from model_id
    if model_id.startswith("huggingface/"):
        model_name = model_id.split("/")[1]
    else:
        model_name = model_id
    
    if model_name not in HUGGINGFACE_MODEL_MAPPING:
        raise ValueError(f"Model {model_id} not supported by HuggingFace provider")
    
    # If LoRA adapter is provided, modify the model_id to indicate it's fine-tuned
    if lora_adapter_path:
        model_id = f"{model_id}_lora"
        print(f"Using LoRA adapter from: {lora_adapter_path}")
    
    results = []
    
    for sample_idx, sample in enumerate(tqdm(samples, desc="Running experiments")):
        # Get model probabilities for the prompt
        response_probs = get_model_probabilities(model_name, sample.prompt, temperature=1.0, lora_adapter_path=lora_adapter_path)
        
        # Sample text to see what the model is actually outputting
        sample_text = sample_model_text(model_name, sample.prompt, max_tokens=10, temperature=1.0, lora_adapter_path=lora_adapter_path)
        
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

def run_experiment_for_model(model_id, num_examples, lora_adapter_path=None):
    """Run experiment for a single model and return results."""
    print(f"Running output control experiment on {model_id}")
    if lora_adapter_path:
        print(f"Using LoRA adapter: {lora_adapter_path}")
    print(f"Parameters: num_examples={num_examples}")
    
    # Validate model availability first
    if not validate_model_availability(model_id, local_only=True):
        print(f"✗ Model {model_id} not available locally.")
        model_name = extract_model_name(model_id)
        print(f"  Please run: python preload_models.py --models {model_name}")
        raise RuntimeError(f"Model {model_id} is not available. Please preload it first.")
    
    # Get samples and run experiment
    samples = get_combined_samples(num_examples=num_examples)
    results = run_experiment(model_id, samples, num_examples, lora_adapter_path)
    
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

