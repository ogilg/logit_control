"""Custom evaluations package - HuggingFace provider functionality."""

from .huggingface_provider import generate_single_token, HUGGINGFACE_MODEL_MAPPING, huggingface_get_probs, huggingface_get_text
from .request_templates import GetTextRequest, GetTextResponse, GetProbsRequest, GetProbsResponse, Message, Prompt

__version__ = "0.1.0"
__all__ = [
    "generate_single_token", 
    "huggingface_get_probs",
    "huggingface_get_text",
    "HUGGINGFACE_MODEL_MAPPING",
    "GetTextRequest",
    "GetTextResponse", 
    "GetProbsRequest",
    "GetProbsResponse",
    "Message",
    "Prompt"
] 