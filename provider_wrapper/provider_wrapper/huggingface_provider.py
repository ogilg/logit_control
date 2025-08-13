"""
HuggingFace provider for evalugator.

Refactored to a class-based design so we can support multiple prompt formats
while sharing model/tokenizer loading, generation and logits logic.
"""

import gc
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

import backoff
import torch
from torch import no_grad, softmax, topk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .data_models import (
    GetProbsRequest,
    GetProbsResponse,
    GetTextRequest,
    GetTextResponse,
    Prompt,
    HUGGINGFACE_MODEL_MAPPING,
    CHAT_MODELS,
)

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
    

# Singleton storage (cached across providers)
_models: Dict[str, Any] = {}
_tokenizers: Dict[str, Any] = {}

def extract_model_name(model_id: str) -> str:
    """Extract the model name from the model_id."""
    return model_id.split("/")[1] if model_id.startswith("huggingface/") else model_id


def provides_model(model_id: str) -> bool:
    """Return True if this provider supports the given model_id."""
    name = extract_model_name(model_id)
    return name in HUGGINGFACE_MODEL_MAPPING


def execute(model_id: str, request):
    """Handle both text generation and probability requests."""
    if not provides_model(model_id):
        raise NotImplementedError(f"Model {model_id} is not supported by HuggingFace provider")
    
    provider = get_provider_for_model(model_id)
    if isinstance(request, GetTextRequest):
        return provider.generate_text(request)
    elif isinstance(request, GetProbsRequest):
        return provider.get_probs(request)
    else:
        raise NotImplementedError(
            f"Request {type(request).__name__} for model {model_id} is not implemented"
        )


class HuggingFaceProvider:
    """Parent provider with shared functionality.

    Child classes should override ``format_prompt`` to return both tokens and text.
    """

    @classmethod
    def get_model_and_tokenizer(
        cls,
        model_id: str,
        lora_adapter_path: str | None = None,
    ):
        """Get or load model and tokenizer using singleton pattern.

        Caches by model_id + quantization + lora-adapter presence so callers can
        share models safely across providers.
        """

        # Cache key includes provider class to allow different loading strategies
        model_key = f"{model_id}_{cls.__name__}_lora" if lora_adapter_path else f"{model_id}_{cls.__name__}"

        if model_key not in _models:
            hf_model_name = HUGGINGFACE_MODEL_MAPPING[model_id]

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Build base kwargs and allow subclass to extend
            model_kwargs: Dict[str, Any] = {
                "device_map": "auto",
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            # Allow subclasses to add provider-specific kwargs (e.g., quantization)
            extra_kwargs = cls._extra_model_kwargs()
            if extra_kwargs:
                model_kwargs.update(extra_kwargs)

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


    def __init__(
        self,
        model_id: str,
        *,
        lora_adapter_path: str | None = None,
    ) -> None:
        self.model_id = extract_model_name(model_id)
        self.lora_adapter_path = lora_adapter_path
        self.model, self.tokenizer = self.get_model_and_tokenizer(
            self.model_id, lora_adapter_path
        )

    @classmethod
    def _extra_model_kwargs(cls) -> Dict[str, Any]:
        """Subclass hook: return extra kwargs for model.from_pretrained.

        Base: no extras.
        """
        return {}

    # ----- Prompt formatting API -----
    def format_prompt(self, prompt: Prompt) -> Tuple[List[int], str]:
        """Return (tokens, text) for the model input.

        Base implementation uses a simple newline-joined prompt.
        """
        text = "\n\n".join([msg.content for msg in prompt])
        tokens = self.tokenizer.encode(text)
        return tokens, text

    # ----- Public helpers -----
    def encode(self, data: str) -> List[int]:
        return self.tokenizer.encode(data)

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    # ----- Core ops -----
    def generate_text(self, request: GetTextRequest) -> GetTextResponse:
        tokens, text = self.format_prompt(request.prompt)

        # Build input tensor from tokens directly for full control
        input_ids = torch.tensor([tokens], dtype=torch.long)
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)

        with no_grad():
            is_greedy = request.temperature is not None and request.temperature <= 0.0
            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": request.max_tokens,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            if is_greedy:
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = request.temperature

            # Optional: Harmony end token if present in this tokenizer
            try:
                harmony_end_id = self.tokenizer.convert_tokens_to_ids("<|end|>")
                if isinstance(harmony_end_id, int) and harmony_end_id >= 0:
                    gen_kwargs["eos_token_id"] = harmony_end_id
            except Exception:
                pass

            outputs = self.model.generate(input_ids=input_ids, **gen_kwargs)

        generated_tokens = outputs[0][input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        if "<|end|>" in generated_text:
            generated_text = generated_text.split("<|end|>", 1)[0]

        return GetTextResponse(
            model_id=self.model_id,
            request=request,
            txt=generated_text,
            raw_responses=[generated_text],
            context={
                "method": "generation",
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "hf_model_name": HUGGINGFACE_MODEL_MAPPING[self.model_id],
                "local_model": True,
                "prompt_text": text,
            },
        )

    def get_probs(self, request: GetProbsRequest) -> GetProbsResponse:
        tokens, text = self.format_prompt(request.prompt)
        input_ids = torch.tensor([tokens], dtype=torch.long)
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)

        with no_grad():
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits[0, -1, :]

        probs_tensor = softmax(logits, dim=-1)
        k = min(50, probs_tensor.shape[-1])
        top_probs, top_indices = topk(probs_tensor, k)

        probs: Dict[str, float] = {}
        for prob, idx in zip(top_probs.detach().cpu().tolist(), top_indices.detach().cpu().tolist()):
            token_str = self.tokenizer.decode([idx])
            probs[token_str] = prob

        return GetProbsResponse(
            model_id=self.model_id,
            request=request,
            probs=probs,
            raw_responses=[probs],
            context={
                "method": "logits_based",
                "top_k": k,
                "hf_model_name": HUGGINGFACE_MODEL_MAPPING[self.model_id],
                "local_model": True,
                "prompt_text": text,
            },
        )


class DefaultHFProvider(HuggingFaceProvider):
    """Default prompt formatting.

    - Chat models: use tokenizer chat template
    - Non-chat models: simple join
    """

    def format_prompt(self, prompt: Prompt) -> Tuple[List[int], str]:
        if self.model_id in CHAT_MODELS:
            messages = [{"role": msg.role, "content": msg.content} for msg in prompt]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = "\n\n".join([msg.content for msg in prompt])
        tokens = self.tokenizer.encode(text)
        return tokens, text

    @classmethod
    def _extra_model_kwargs(cls) -> Dict[str, Any]:
        # Apply 4-bit quantization by default for standard HF models
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            ),
            "torch_dtype": torch.bfloat16,
        }


class GPTOSSProvider(HuggingFaceProvider):
    """GPT-OSS prompt formatting via openai-harmony library."""

    def format_prompt(self, prompt: Prompt) -> Tuple[List[int], str]:
        enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

        # System message declaring channels and reasoning effort
        sys_content = (
            SystemContent.new()
            .with_required_channels(["analysis", "commentary", "final"])
            .with_reasoning_effort("medium")
        )

        msgs: List[HarmonyMessage] = []
        msgs.append(HarmonyMessage.from_role_and_content(HarmonyRole.SYSTEM, sys_content))

        for msg in prompt:
            role = (msg.role or "user").lower()
            if role == "user":
                msgs.append(HarmonyMessage.from_role_and_content(HarmonyRole.USER, msg.content))
            elif role == "assistant":
                msgs.append(
                    HarmonyMessage.from_role_and_content(HarmonyRole.ASSISTANT, msg.content)
                )
            elif role == "system":
                msgs.append(HarmonyMessage.from_role_and_content(HarmonyRole.SYSTEM, msg.content))
            else:
                msgs.append(HarmonyMessage.from_role_and_content(HarmonyRole.USER, msg.content))

        # Open the target channel for the assistant
        msgs.append(
            HarmonyMessage.from_role_and_content(HarmonyRole.ASSISTANT, "").with_channel("final")
        )

        convo = HarmonyConversation.from_messages(msgs)
        token_list: List[int] = enc.render_conversation_for_completion(convo, HarmonyRole.ASSISTANT)

        # Best-effort: create a human-readable text by decoding with HF tokenizer
        # (not guaranteed to be identical to the wire-format, but helpful for logging)
        text = self.tokenizer.decode(token_list)
        return token_list, text

    def parse_completion(self, output_tokens: List[int]):
        """Parse completion tokens using Harmony encoding into structured messages.

        Returns whatever the Harmony library yields from
        encoding.parse_messages_from_completion_tokens(new_tokens, Role.ASSISTANT).
        """
        enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        parsed = enc.parse_messages_from_completion_tokens(output_tokens, HarmonyRole.ASSISTANT)
        return parsed


# ----- Dummy helpers for lightweight tests -----
def dummy_get_text(model_id: str, request: GetTextRequest) -> GetTextResponse:
    """Return a lightweight dummy text response with correct shape.

    - For GPT-OSS, include a minimal Harmony-parsed structure in context.
    - For default models, return plain text only.
    """
    provider = get_provider_for_model(model_id, request=request)
    txt = "Hi!"

    context: Dict[str, Any] = {
        "method": "dummy_generation",
        "hf_model_name": HUGGINGFACE_MODEL_MAPPING[extract_model_name(model_id)],
        "local_model": False,
    }

    if isinstance(provider, GPTOSSProvider):
        # Create a minimal parsed messages stub via Harmony: a single assistant/final message
        enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        parsed = [{"role": "assistant", "content": txt, "channel": "final"}]
        context.update({
            "harmony": True,
            "parsed_messages": parsed,
        })
    else:
        context["harmony"] = False

    return GetTextResponse(
        model_id=extract_model_name(model_id),
        request=request,
        txt=txt,
        raw_responses=[txt],
        context=context,
    )


def dummy_get_probs(model_id: str, request: GetProbsRequest) -> GetProbsResponse:
    """Return a lightweight dummy top-k probability dict with correct shape.

    - For GPT-OSS, flag harmony=True in context.
    """
    provider = get_provider_for_model(model_id, request=request)
    probs = {"Hello": 0.6, "World": 0.4}
    context: Dict[str, Any] = {
        "method": "dummy_logits",
        "top_k": 2,
        "hf_model_name": HUGGINGFACE_MODEL_MAPPING[extract_model_name(model_id)],
        "local_model": False,
        "harmony": isinstance(provider, GPTOSSProvider),
    }
    return GetProbsResponse(
        model_id=extract_model_name(model_id),
        request=request,
        probs=probs,
        raw_responses=[probs],
        context=context,
    )


def get_provider_for_model(
    model_id: str,
    request: Any | None = None,
    *,
    lora_adapter_path: str | None = None,
) -> HuggingFaceProvider:
    """Factory to return the appropriate provider instance for a model_id.

    Respects optional lora/quantization hints from request.context if present.
    """
    name = extract_model_name(model_id)
    # Prefer explicit args; fall back to request context
    if lora_adapter_path is None and request is not None:
        lora_adapter_path = getattr(request, "lora_adapter_path", None)

    if name == "gpt-oss-20b":
        return GPTOSSProvider(name, lora_adapter_path=lora_adapter_path)
    return DefaultHFProvider(name, lora_adapter_path=lora_adapter_path)



# Model Management Functions

def preload_model(model_id: str, verbose: bool = False) -> bool:
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
        # Use the factory to choose the right provider class and trigger load
        provider = get_provider_for_model(model_name)
        _ = provider.model  # ensure loaded
        
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
        AutoTokenizer.from_pretrained(hf_model_name, local_files_only=local_only)
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
        keys_to_remove = [key for key in list(_models.keys()) if key.startswith(model_name)]
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


# ----- Simple module-level helpers for compatibility -----
def encode(model_id: str, data: str) -> List[int]:
    provider = DefaultHFProvider(extract_model_name(model_id))
    return provider.encode(data)


def decode(model_id: str, tokens: List[int]) -> str:
    provider = DefaultHFProvider(extract_model_name(model_id))
    return provider.decode(tokens)


