"""
HuggingFace provider for evalugator.

Supports DeepSeek, Qwen, Mistral, and Llama models.
Uses local GPU models for both text generation and probability extraction.
"""

from typing import Any, Dict

import backoff
import torch
import torch.nn.functional as F
from torch import no_grad, softmax, topk
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .data_models import (GetProbsRequest, GetProbsResponse,
                          GetTextRequest, GetTextResponse, Prompt,
                          HUGGINGFACE_MODEL_MAPPING, CHAT_MODELS)


def on_backoff(details):
    print(f"Repeating failed request (attempt {details['tries']}). Reason: {details['exception']}")
    


# Singleton storage
_models: Dict[str, Any] = {}
_tokenizers: Dict[str, Any] = {}


def provides_model(model_id: str) -> bool:
    """Return True if this provider supports the given model_id."""
    if model_id.startswith("huggingface/"):
        model_id = model_id.split("/")[1]
        return model_id in HUGGINGFACE_MODEL_MAPPING
    return False


def execute(model_id: str, request):
    """Handle both text generation and probability requests."""
    if not provides_model(model_id):
        raise NotImplementedError(f"Model {model_id} is not supported by HuggingFace provider")
    
    # Extract actual model_id if prefixed (same pattern as replicate)
    model_name = model_id.split("/")[1] if model_id.startswith("huggingface/") else model_id
    
    if isinstance(request, GetTextRequest):
        return huggingface_get_text(model_name, request)
    elif isinstance(request, GetProbsRequest):
        return huggingface_get_probs(model_name, request)
    else:
        raise NotImplementedError(
            f"Request {type(request).__name__} for model {model_id} is not implemented"
        )


def encode(model_id: str, data: str) -> list[int]:
    """Convert text to tokens - requires local model."""
    # Extract actual model_id if prefixed (same pattern as replicate)
    model_name = model_id.split("/")[1] if model_id.startswith("huggingface/") else model_id
    
    tokenizer = _get_tokenizer(model_name)
    return tokenizer.encode(data)


def decode(model_id: str, tokens: list[int]) -> str:
    """Convert tokens to text - requires local model."""
    # Extract actual model_id if prefixed (same pattern as replicate)
    model_name = model_id.split("/")[1] if model_id.startswith("huggingface/") else model_id
    
    tokenizer = _get_tokenizer(model_name)
    return tokenizer.decode(tokens, skip_special_tokens=True)


def _get_model_and_tokenizer(model_id: str, lora_adapter_path: str = None):
    """Get or load model and tokenizer using singleton pattern."""
    
    # Create a unique key for the model with LoRA adapter
    model_key = f"{model_id}_lora" if lora_adapter_path else model_id
    
    if model_key not in _models:
        hf_model_name = HUGGINGFACE_MODEL_MAPPING[model_id]
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # Load model with quantization - let transformers handle device placement
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,  
            quantization_config=bnb_config,
            device_map="auto",  # Let transformers automatically handle device placement
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            resume_download=True,
        )
        
        # Load LoRA adapter if provided
        if lora_adapter_path:
            try:
                from peft import PeftModel
                print(f"Loading LoRA adapter from {lora_adapter_path}")
                model = PeftModel.from_pretrained(model, lora_adapter_path)
                print("LoRA adapter loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load LoRA adapter from {lora_adapter_path}: {e}")
                print("Continuing with base model")
        
        _models[model_key] = model
        _tokenizers[model_key] = tokenizer
    
    return _models[model_id], _tokenizers[model_id]


def _get_tokenizer(model_id: str):
    """Get tokenizer for the given model."""
    if model_id not in _tokenizers:
        _get_model_and_tokenizer(model_id)  # This will load both
    return _tokenizers[model_id]


def _format_prompt(prompt: Prompt, model_id: str, tokenizer) -> str:
    """Format prompt for the specific model."""
    if model_id in CHAT_MODELS:
        # Use chat template for chat models
        messages = [{"role": msg.role, "content": msg.content} for msg in prompt]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Use simple concatenation for non-chat models
        return "\n\n".join([msg.content for msg in prompt])


@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(RuntimeError,),
    max_value=60,
    factor=1.5,
    on_backoff=on_backoff,
)
def huggingface_get_text(model_id: str, request: GetTextRequest) -> GetTextResponse:
    """Generate text using local GPU model."""
    
    # Check if LoRA adapter path is provided in the request context
    lora_adapter_path = getattr(request, 'lora_adapter_path', None)
    
    model, tokenizer = _get_model_and_tokenizer(model_id, lora_adapter_path)
    
    # Format the prompt
    formatted_prompt = _format_prompt(request.prompt, model_id, tokenizer)
    
    # Tokenize the input
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    # Move to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate text
    with no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the generated text
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return GetTextResponse(
        model_id=model_id,
        request=request,
        txt=generated_text,
        raw_responses=[generated_text],
        context={
            "method": "generation",
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "hf_model_name": HUGGINGFACE_MODEL_MAPPING[model_id],
            "local_model": True,
        }
    )


@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(RuntimeError,),
    max_value=60,
    factor=1.5,
    on_backoff=on_backoff,
)
def huggingface_get_probs(model_id: str, request: GetProbsRequest) -> GetProbsResponse:
    """Get probability distribution for next token using local model logits."""
    
    # Check if LoRA adapter path is provided in the request context
    lora_adapter_path = getattr(request, 'lora_adapter_path', None)
    
    model, tokenizer = _get_model_and_tokenizer(model_id, lora_adapter_path)
    
    # Format prompt
    prompt_text = _format_prompt(request.prompt, model_id, tokenizer)
    
    # Tokenize input
    inputs = tokenizer(prompt_text, return_tensors="pt")
    
    # Move to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get logits for the next token
    with no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Last token's logits
    
    # Convert to probabilities
    probs_tensor = softmax(logits, dim=-1)
    
    # Get top-k tokens (default to 50 as mentioned in requirements)
    k = min(50, len(probs_tensor))
    top_probs, top_indices = topk(probs_tensor, k)
    
    # Convert to dictionary
    probs = {}
    for prob, idx in zip(top_probs.cpu().tolist(), top_indices.cpu().tolist()):
        token = tokenizer.decode([idx])
        probs[token] = prob
    
    return GetProbsResponse(
        model_id=model_id,
        request=request,
        probs=probs,
        raw_responses=[probs],
        context={
            "method": "logits_based",
            "top_k": k,
            "hf_model_name": HUGGINGFACE_MODEL_MAPPING[model_id],
            "local_model": True,
        }
    )
