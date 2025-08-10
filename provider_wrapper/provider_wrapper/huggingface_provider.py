"""
HuggingFace provider for evalugator.

Supports DeepSeek, Qwen, Mistral, and Llama models.
Uses local GPU models for both text generation and probability extraction.
"""

import gc
import os
import shutil
from typing import Any, Dict, List, Optional

import backoff
import torch
import torch.nn.functional as F
from torch import no_grad, softmax, topk
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)

from .data_models import (GetProbsRequest, GetProbsResponse,
                          GetTextRequest, GetTextResponse, Prompt,
                          HUGGINGFACE_MODEL_MAPPING, CHAT_MODELS)

from openai_harmony import (
            load_harmony_encoding,
            HarmonyEncodingName,
            Role as HarmonyRole,
            Message as HarmonyMessage,
            Conversation as HarmonyConversation,
            SystemContent,
        )


def on_backoff(details):
    print(f"Repeating failed request (attempt {details['tries']}). Reason: {details['exception']}")
    

# Singleton storage
_models: Dict[str, Any] = {}
_tokenizers: Dict[str, Any] = {}

def extract_model_name(model_id: str) -> str:
    """Extract the model name from the model_id."""
    return model_id.split("/")[1] if model_id.startswith("huggingface/") else model_id


def provides_model(model_id: str) -> bool:
    """Return True if this provider supports the given model_id."""
    if model_id.startswith("huggingface/"):
        model_id = extract_model_name(model_id)
        return model_id in HUGGINGFACE_MODEL_MAPPING
    return False


def execute(model_id: str, request):
    """Handle both text generation and probability requests."""
    if not provides_model(model_id):
        raise NotImplementedError(f"Model {model_id} is not supported by HuggingFace provider")
    
    # Extract actual model_id if prefixed (same pattern as replicate)
    model_name = extract_model_name(model_id)
    
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
    model_name = extract_model_name(model_id)
    
    tokenizer = _get_tokenizer(model_name)
    return tokenizer.encode(data)


def decode(model_id: str, tokens: list[int]) -> str:
    """Convert tokens to text - requires local model."""
    # Extract actual model_id if prefixed (same pattern as replicate)
    model_name = extract_model_name(model_id)
    
    tokenizer = _get_tokenizer(model_name)
    return tokenizer.decode(tokens, skip_special_tokens=True)


def _get_model_and_tokenizer(
    model_id: str,
    lora_adapter_path: str | None = None,
    quantization: str = "bnb",
):
    """Get or load model and tokenizer using singleton pattern."""
    
    # Create a unique key for the model with LoRA adapter and quantization mode
    model_key = f"{model_id}_{quantization}_lora" if lora_adapter_path else f"{model_id}_{quantization}"
    
    if model_key not in _models:
        hf_model_name = HUGGINGFACE_MODEL_MAPPING[model_id]
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        

        # Build a single kwargs dict and make one from_pretrained call
        model_kwargs: Dict[str, Any] = {
            "device_map": "auto",
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if quantization == "bnb":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["torch_dtype"] = torch.bfloat16
        # mxfp4: do not pass quantization_config; let repo load native MXFP4

        model = AutoModelForCausalLM.from_pretrained(hf_model_name, **model_kwargs)
        
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
    
    return _models[model_key], _tokenizers[model_key]


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


# def _format_prompt_harmony(prompt: Prompt, tokenizer) -> str:
#     """Render messages in OpenAI Harmony format for gpt-oss models.

#     We explicitly open an assistant message on the `final` channel for the next turn
#     and DO NOT close it, so generation continues inside the final channel.
#     """
#     # Minimal system message declaring valid channels per Harmony spec
#     rendered: list[str] = [
#         "<|start|>system<|message|># Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>",
#     ]

#     # Render all but the last message as closed messages
#     if len(prompt) > 0:
#         for msg in prompt[:-1]:
#             role = msg.role
#             content = msg.content
#             if role == "assistant":
#                 rendered.append(f"<|start|>assistant<|channel|>final<|message|>{content}<|end|>")
#             elif role == "user":
#                 rendered.append(f"<|start|>user<|message|>{content}<|end|>")
#             elif role == "system":
#                 rendered.append(f"<|start|>system<|message|>{content}<|end|>")
#             else:
#                 # Fallback to user role if unknown
#                 rendered.append(f"<|start|>user<|message|>{content}<|end|>")

#     # For the last message, we open an assistant final message to continue generation
#     if len(prompt) == 0:
#         # No prior messages – just open final channel
#         rendered.append("<|start|>assistant<|channel|>final<|message|>")
#     else:
#         last = prompt[-1]
#         # We always continue inside assistant/final; include any assistant stub text
#         stub_text = last.content if last.role == "assistant" else ""
#         # If the last is a user message, include it as a closed message first
#         if last.role == "user":
#             rendered.append(f"<|start|>user<|message|>{last.content}<|end|>")
#             stub_text = ""
#         rendered.append(f"<|start|>assistant<|channel|>final<|message|>{stub_text}")

#     return "".join(rendered)


def _format_prompt_harmony_lib(prompt: Prompt) -> str | None:
    """Use the openai-harmony PyPI package to render the conversation for completion.

    Returns a rendered string if possible; otherwise returns None to signal fallback.
    """
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    msgs = []

    # Add a minimal system message so encoding inserts channel metadata
    msgs.append(HarmonyMessage.from_role_and_content(HarmonyRole.SYSTEM, SystemContent.new()))

    for msg in prompt:
        role = msg.role.lower()
        if role == "user":
            msgs.append(HarmonyMessage.from_role_and_content(HarmonyRole.USER, msg.content))
        elif role == "assistant":
            msgs.append(HarmonyMessage.from_role_and_content(HarmonyRole.ASSISTANT, msg.content))
        elif role == "system":
            # Additional system text can be appended as plain string
            msgs.append(HarmonyMessage.from_role_and_content(HarmonyRole.SYSTEM, msg.content))
        else:
            msgs.append(HarmonyMessage.from_role_and_content(HarmonyRole.USER, msg.content))

    convo = HarmonyConversation.from_messages(msgs)

    # Prefer a text-rendering API if available
    render_text = getattr(enc, "render_conversation_for_completion_as_text", None)
    if callable(render_text):
        rendered_text = render_text(convo, HarmonyRole.ASSISTANT)
        # Ensure next completion is in assistant/final channel
        if isinstance(rendered_text, str):
            last_asst = rendered_text.rfind("<|start|>assistant")
            if last_asst != -1:
                tail = rendered_text[last_asst:]
                if "<|channel|>" not in tail:
                    rendered_text = rendered_text + "<|channel|>final<|message|>"
        return rendered_text

    rendered = enc.render_conversation_for_completion(convo, HarmonyRole.ASSISTANT)
    # If this already is a string, return it
    if isinstance(rendered, str):
        s = rendered
        last_asst = s.rfind("<|start|>assistant")
        if last_asst != -1:
            tail = s[last_asst:]
            if "<|channel|>" not in tail:
                s = s + "<|channel|>final<|message|>"
        return s

    # Try to convert token sequence to text if the encoder exposes a method
    tokens_to_text = getattr(enc, "tokens_to_text", None)
    if callable(tokens_to_text):
        s = tokens_to_text(rendered)
        if isinstance(s, str):
            last_asst = s.rfind("<|start|>assistant")
            if last_asst != -1:
                tail = s[last_asst:]
                if "<|channel|>" not in tail:
                    s = s + "<|channel|>final<|message|>"
        return s

    # Could not convert to text
    return None


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
    
    quantization = getattr(request, "context", {}).get("quantization", None) if hasattr(request, "context") and isinstance(request.context, dict) else None
    model, tokenizer = _get_model_and_tokenizer(model_id, lora_adapter_path, quantization)
    
    # Format the prompt
    if model_id == "gpt-oss-20b":
        # Prefer the official Harmony library; fall back to minimal renderer
        formatted_prompt = _format_prompt_harmony_lib(request.prompt)
    else:
        formatted_prompt = _format_prompt(request.prompt, model_id, tokenizer)
    
    # Tokenize the input
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    # Move to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate text
    with no_grad():
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": request.max_tokens,
            "temperature": request.temperature,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
        }
        # If Harmony end token exists, use it as eos
        try:
            harmony_end_id = tokenizer.convert_tokens_to_ids("<|end|>")
            if isinstance(harmony_end_id, int) and harmony_end_id >= 0:
                gen_kwargs["eos_token_id"] = harmony_end_id
        except Exception:
            pass

        outputs = model.generate(
            **inputs,
            **gen_kwargs,
        )
    
    # Decode the generated text
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    # If the Harmony end marker leaked as plain text, truncate at it
    end_marker = "<|end|>"
    if end_marker in generated_text:
        generated_text = generated_text.split(end_marker, 1)[0]
    
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
    
    quantization = getattr(request, "context", {}).get("quantization", None) if hasattr(request, "context") and isinstance(request.context, dict) else None
    model, tokenizer = _get_model_and_tokenizer(model_id, lora_adapter_path, quantization)
    
    # Format prompt
    if model_id == "gpt-oss-20b":
        # Align with generation: next token should be inside assistant/final
        prompt_text = _format_prompt_harmony_lib(request.prompt)
    else:
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


# Model Management Functions

def preload_model(model_id: str, verbose: bool = False, quantization: str | None = None) -> bool:
    """
    Preload a model into the HuggingFace cache.
    
    Args:
        model_id: Model identifier (e.g., "llama-3.1-8b")
        verbose: If True, print detailed loading messages
        
    Returns:
        True if successful, False otherwise
    """
    
    # Extract actual model name
    model_name = extract_model_name(model_id)
    
    if model_name not in HUGGINGFACE_MODEL_MAPPING:
        if verbose:
            print(f"ERROR: Model {model_id} not found in HUGGINGFACE_MODEL_MAPPING")
        return False
    
    hf_model_name = HUGGINGFACE_MODEL_MAPPING[model_name]
    if verbose:
        print(f"Preloading {model_id} ({hf_model_name})")
    
    try:
        # Use the existing _get_model_and_tokenizer function which handles caching
        model, tokenizer = _get_model_and_tokenizer(model_name, None, quantization)
        
        if verbose:
            print(f"✓ {model_id} preloaded successfully")
        
        # DO NOT clear from memory - keep it cached for reuse
        # Only clear GPU memory to free up space for other operations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"✗ Failed to preload {model_id}: {e}")
        return False


def validate_model_availability(model_id: str, local_only: bool = True) -> bool:
    """
    Check if a model is available (downloaded) without loading it fully.
    
    Args:
        model_id: Model identifier
        local_only: If True, only check local cache (don't download)
        
    Returns:
        True if model is available, False otherwise
    """
    
    # Extract actual model name
    model_name = extract_model_name(model_id)
    
    if model_name not in HUGGINGFACE_MODEL_MAPPING:
        return False
    
    hf_model_name = HUGGINGFACE_MODEL_MAPPING[model_name]
    
    try:
        # Try to load tokenizer (lightweight check)
        tokenizer = AutoTokenizer.from_pretrained(
            hf_model_name, 
            local_files_only=local_only
        )
        return True
    except Exception:
        return False


def clear_model_cache(model_id: Optional[str] = None):
    """
    Clear models from memory cache.
    
    Args:
        model_id: Specific model to clear, or None to clear all
    """
    
    if model_id is None:
        # Clear all models
        _models.clear()
        _tokenizers.clear()
    else:
        # Clear specific model
        model_name = extract_model_name(model_id)
        
        # Remove from singleton dictionaries
        keys_to_remove = [key for key in _models.keys() if key.startswith(model_name)]
        for key in keys_to_remove:
            if key in _models:
                del _models[key]
            if key in _tokenizers:
                del _tokenizers[key]
    
    # Clear GPU memory
    clear_gpu_memory()


def clear_huggingface_disk_cache():
    """Clear HuggingFace disk cache to free space."""
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


def clear_gpu_memory():
    """Clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


