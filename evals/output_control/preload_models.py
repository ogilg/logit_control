#!/usr/bin/env python3
"""
Script to preload a single model into HuggingFace cache.
This separates model downloading from evaluation to handle download failures gracefully.
"""

import argparse
import sys

from provider_wrapper import preload_model


def main():
    parser = argparse.ArgumentParser(description="Preload a single model into HuggingFace cache")
    parser.add_argument("model", 
                       help="Model ID to preload")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output during preloading")
    
    args = parser.parse_args()
    
    # Preload the single model
    print("\n" + "=" * 60)
    print("PRELOADING MODEL")
    print("=" * 60)
    
    print(f"Preloading {args.model}...")
    success = preload_model(args.model, verbose=args.verbose)
    
    # Print summary
    print("\n" + "=" * 60)
    print("PRELOAD SUMMARY")
    print("=" * 60)
    
    if success:
        print(f"✓ Successfully preloaded: {args.model}")
        print(f"\nModel is ready! You can now run evaluations.")
        return 0
    else:
        print(f"✗ Failed to preload: {args.model}")
        print(f"\nYou may want to retry or check your internet connection.")
        return 1


if __name__ == "__main__":
    sys.exit(main())