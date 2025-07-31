"""
Data models for custom evaluations package.

This module contains data structures used throughout the custom-evals package.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from .request_templates import Prompt


@dataclass
class Sample:
    """A sample for training or evaluation with prompt and target distribution."""
    prompt: List[Prompt]
    target_distribution: Dict[str, float]
    word1: str
    word2: str
    tvd_tolerance: float
    is_given_words: bool
    seed: Optional[int] = None
    case_type: str = "given_words"  # Default case type 