"""
Data models for custom evaluations package.

This module contains data structures used throughout the custom-evals package.
"""

from abc import abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


from .request_templates import Prompt

# Model mapping from evalugator model IDs to HuggingFace model names
HUGGINGFACE_MODEL_MAPPING = {
    # DeepSeek
    "deepseek-coder-7b": "deepseek-ai/deepseek-coder-7b-base",
    "deepseek-coder-7b-instruct": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
    "deepseek-coder-33b": "deepseek-ai/deepseek-coder-33b-base",
    "deepseek-coder-33b-instruct": "deepseek-ai/deepseek-coder-33b-instruct-v1.5",
    "deepseek-llm-7b": "deepseek-ai/deepseek-llm-7b-base",
    "deepseek-llm-7b-chat": "deepseek-ai/deepseek-llm-7b-chat",

    # Qwen
    "qwen2.5-7b": "Qwen/Qwen2-7B",
    "qwen2.5-7b-instruct": "Qwen/Qwen2-7B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2-3B",
    "qwen2.5-3b-instruct": "Qwen/Qwen2-3B-Instruct",
    "qwen1.5-7b": "Qwen/Qwen1.5-7B",
    "qwen1.5-7b-chat": "Qwen/Qwen1.5-7B-Chat",
    "qwen1.5-14b": "Qwen/Qwen1.5-14B",
    "qwen1.5-14b-chat": "Qwen/Qwen1.5-14B-Chat",

    # Mistral
    "mistral-7b": "mistralai/Mistral-7B-v0.3",
    "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-v0.1",
    "mixtral-8x7b-instruct": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistral-small": "mistralai/Mistral-small",
    "mistral-medium": "mistralai/Mistral-medium",

    # Llama
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "llama-2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama-2-13b": "meta-llama/Llama-2-13b-hf",
    "llama-2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    "llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",

    # Phi
    "phi-2": "microsoft/phi-2",
    "phi-3-mini-4k-instruct": "microsoft/phi-3-mini-4k-instruct",
    "phi-3-mini-128k-instruct": "microsoft/phi-3-mini-128k-instruct",
    "phi-3-small-8k-instruct": "microsoft/phi-3-small-8k-instruct",

    # Gemma
    "gemma-2b": "google/gemma-2b",
    "gemma-2b-it": "google/gemma-2b-it",
    "gemma-7b": "google/gemma-7b",
    "gemma-7b-it": "google/gemma-7b-it",
}

# Models that support chat format (instruct/chat models)
CHAT_MODELS = {
    "deepseek-coder-7b-instruct",
    "deepseek-coder-33b-instruct",
    "deepseek-llm-7b-chat",
    "qwen2.5-7b-instruct",
    "qwen2.5-3b-instruct",
    "qwen1.5-7b-chat",
    "qwen1.5-3b-chat",
    "qwen1.5-14b-chat",
    "mistral-7b-instruct",
    "mixtral-8x7b-instruct",
    "mistral-small",
    "mistral-medium",
    "llama-2-7b-chat",
    "llama-2-13b-chat",
    "llama-3.1-8b-instruct",
    "llama-3.2-3b-instruct",
    "phi-3-mini-4k-instruct",
    "phi-3-mini-128k-instruct",
    "phi-3-small-8k-instruct",
    "gemma-2b-it",
    "gemma-7b-it",
}


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


@dataclass
class Message:
    role: str
    content: str


Prompt = list[Message]


@dataclass(kw_only=True)
class Request:
    #   Request context stores information that should be consider private/internal
    #   and providers should not access it. The structure might be different for different evals.
    context: Any
    prompt: Prompt

    def as_dict(self):
        return asdict(self)


@dataclass(kw_only=True)
class Response:
    model_id: str
    request: Request

    #   Raw responses from the API should go here. This is intended mostly for debuging.
    raw_responses: list[Any]

    #   Response context is meant mostly for low-level logging. 
    #   Provider is allowed to put anything (json-serializable) here, e.g.:
    #   *   exact prompt/messages structure sent to the API
    #   *   execution time
    #   *   additional response data (logprobs or whatever)
    #   *   provider config (if there's a config)
    #   It's also perfectly fine to set it to None.
    #
    #   NOTE: In some contexts (e.g. in solvers), evalugator translates some responses to different
    #         responses, so the response returned by the provider will not necessarely be the same
    #         as the response returned to the eval code.
    context: Any

    @abstractmethod
    def as_dict(self):
        raise NotImplementedError


@dataclass(kw_only=True)
class GetTextRequest(Request):
    temperature: float
    max_tokens: int


@dataclass(kw_only=True)
class GetTextResponse(Response):
    txt: str

    def as_dict(self):
        return {
            "model_id": self.model_id,
            "request": self.request.as_dict(),
            "txt": self.txt,
            "context": self.context,
        }


@dataclass(kw_only=True)
class GetProbsRequest(Request):
    #   If you need only top N most likely tokens, set this value.
    #   Provider is allowed to return more tokens than this.
    #   The only effect (right now) is that with this value set to <=5, 
    #   openai uses logprobs, and with a higher value raises NotImplementedError.
    min_top_n: int

    #   If this is not None, providers that don't support logprobs will
    #   do sampling with temperature=1
    num_samples: int | None


@dataclass(kw_only=True)
class GetProbsResponse(Response):
    probs: dict[str, float]

    def as_dict(self):
        return {
            "model_id": self.model_id,
            "request": self.request.as_dict(),
            "probs": self.probs,
            "context": self.context,
        }
