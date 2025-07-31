"""Custom evaluations package - HuggingFace provider functionality with LoRA fine-tuning support."""

from .huggingface_provider import generate_single_token, HUGGINGFACE_MODEL_MAPPING, huggingface_get_probs, huggingface_get_text
from .request_templates import GetTextRequest, GetTextResponse, GetProbsRequest, GetProbsResponse, Message, Prompt
from .data_models import Sample
from ..lora.lora_config import LoRAConfig
from ..lora.lora_finetuner import LoRAFineTuner
from ..lora.loss_functions import CustomLossFunction, TVDLoss, KLDivergenceLoss

__version__ = "0.1.0"
__all__ = [
    # HuggingFace provider
    "generate_single_token", 
    "huggingface_get_probs",
    "huggingface_get_text",
    "HUGGINGFACE_MODEL_MAPPING",
    
    # Request templates
    "GetTextRequest",
    "GetTextResponse", 
    "GetProbsRequest",
    "GetProbsResponse",
    "Message",
    "Prompt",
    
    # Data models
    "Sample",
    
    # LoRA configuration
    "LoRAConfig",
    
    # LoRA fine-tuning
    "LoRAFineTuner",
    "CustomLossFunction",
    "TVDLoss",
    "KLDivergenceLoss"
] 