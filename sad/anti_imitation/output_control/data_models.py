from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Prompt:
    role: str
    content: str


@dataclass
class Sample:
    prompt: List[Prompt]
    target_distribution: Dict[str, float]
    word1: str
    word2: str
    tvd_tolerance: float
    is_given_words: bool
    seed: int = None
    case_type: str = "given_words"  # Default case type 