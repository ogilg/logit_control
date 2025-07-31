"""Custom evaluations package - HuggingFace provider functionality with LoRA fine-tuning support."""

from .lora_config import LoRAConfig
from .lora_finetuner import LoRAFineTuner
from .loss_functions import CustomLossFunction, KLDivergenceLoss, TVDLoss
from .data_models import Sample
from .huggingface_provider import (HUGGINGFACE_MODEL_MAPPING,
                                   generate_single_token,
                                   huggingface_get_probs, huggingface_get_text)
from .request_templates import (GetProbsRequest, GetProbsResponse,
                                GetTextRequest, GetTextResponse, Message,
                                Prompt)

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