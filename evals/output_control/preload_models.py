#!/usr/bin/env python3
"""
Script to preload models into HuggingFace cache.
This separates model downloading from evaluation to handle download failures gracefully.
"""

import argparse
import sys
from typing import List

from provider_wrapper import preload_models_batch


def main():
    parser = argparse.ArgumentParser(description="Preload models into HuggingFace cache")
    parser.add_argument("--models", nargs="+", 
                       help="List of model IDs to preload. If not provided, uses default model list.")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output during preloading")
    
    args = parser.parse_args()
    
    # Default model list (same as run_experiment.py)
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
    
    models_to_preload = args.models if args.models else default_models
    
    # Preload all models using provider_wrapper
    results = preload_models_batch(models_to_preload, verbose=args.verbose)
    
    # Print summary
    print("\n" + "=" * 60)
    print("PRELOAD SUMMARY")
    print("=" * 60)
    
    successful = [model for model, success in results.items() if success]
    failed = [model for model, success in results.items() if not success]
    
    print(f"Successfully preloaded: {len(successful)}/{len(results)} models")
    
    if successful:
        print("\n✓ Ready for evaluation:")
        for model in successful:
            print(f"  - {model}")
    
    if failed:
        print("\n✗ Failed to preload:")
        for model in failed:
            print(f"  - {model}")
        print(f"\nYou may want to retry these models or check your internet connection.")
        return 1
    
    print(f"\nAll models are ready! You can now run evaluations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())