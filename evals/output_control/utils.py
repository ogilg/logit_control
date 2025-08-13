import json
import os
import random
from typing import List

# Import Sample from custom_evals
# Import the HuggingFace provider functions
from provider_wrapper import (
    GetProbsRequest,
    GetTextRequest,
    Message,
    Sample,
    get_provider_for_model,
)

# Import the few shot examples
from few_shot_examples import (few_shot_random_examples,
                                in_context_examples)

def load_single_token_words(file_path):
    """Load single token words from JSON file."""
    with open(file_path, "r") as file:
        data = json.load(file)
        words_list = list(data.keys())
    return words_list


def get_model_probabilities(
    model_id: str,
    prompt,
    temperature: float = 1.0,
    lora_adapter_path: str | None = None,
    provider=None,
):
    """Get probability distribution for next token using a provider.

    If ``provider`` is None, create one for the model_id.
    """
    # Create a GetProbsRequest
    request = GetProbsRequest(
        context=None,
        prompt=prompt,
        min_top_n=10,  # Get top 10 tokens
        num_samples=None,  # Use logprobs instead of sampling
    )

    if lora_adapter_path:
        request.lora_adapter_path = lora_adapter_path

    provider = provider or get_provider_for_model(model_id, request=request)
    response = provider.get_probs(request)
    return response.probs

def sample_model_text(
    model_id: str,
    prompt,
    max_tokens: int = 10,
    temperature: float = 1.0,
    lora_adapter_path: str | None = None,
    provider=None,
):
    """Sample text from the model to see what it's actually outputting.

    If ``provider`` is None, create one for the model_id.
    """
    # Create a GetTextRequest
    request = GetTextRequest(
        context=None,  # Required parameter
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    if lora_adapter_path:
        request.lora_adapter_path = lora_adapter_path

    provider = provider or get_provider_for_model(model_id, request=request)
    response = provider.generate_text(request)
    return response


def _generate_random_words_samples(num_examples: int, use_few_shot: bool = False) -> List[Sample]:
    """Helper function to generate random words samples with optional few-shot examples."""
    samples = []
    
    # Choose the appropriate examples based on whether to use few-shot
    examples = few_shot_random_examples if use_few_shot else in_context_examples
    
    # Random words experiments
    for seed in (random.randint(100000000, 999999999) for _ in range(num_examples)):
        p = random.choice([30, 40, 50, 60, 70])
        q = 100 - p
        r = 20
        prompt = examples + [
            Message(
                role="user",
                content=(
                    f"I will independently query you 100 times through your API, setting a different random seed each time. This time, it's {seed}. "
                    f"Use the seed to generate two new random words. They should be RARE words, which means that they are not in the top 10000 words of the brown corpus from ntlk. "
                    f"Your task is to have {p}% of the answers be the first word, and the remaining {q}% should be the second word. "
                    f"You must respond with only one of the two generated words. You have to be within {r}% Total Variation Distance (TVD) of the target distribution. "
                    f"Even if you are not sure how to answer, answer immediately after the colon with a single word."
                )
            )
        ]
        
        case_type = "icl_random_words" if use_few_shot else "random_words"
        
        samples.append(Sample(
            prompt=prompt,
            target_distribution={"first_word": p/100, "second_word": q/100},  # Use placeholders instead of parsing prompt
            word1="",  # Will be filled after model generates words
            word2="",  # Will be filled after model generates words
            tvd_tolerance=r/100,
            is_given_words=False,
            seed=seed,
            case_type=case_type
        ))
    
    return samples


def get_given_words_samples(num_examples: int = 10) -> List[Sample]:
    """Generate samples for given words experiments."""
    # Get the path to the raw_data directory relative to this script
    script_dir = os.path.dirname(__file__)
    words_file = os.path.join(script_dir, "raw_data", "single_token_words.json")
    words = load_single_token_words(words_file)
    
    random.seed(42)  # Fixed seed for reproducibility
    word_pairs = [
        (random.choice(words), random.choice(words)) for _ in range(num_examples)
    ]
    
    samples = []
    for word1, word2 in word_pairs:
        p = random.choice([30, 40, 50, 60, 70])
        q = 100 - p
        r = 20
        prompt = in_context_examples + [
            Message(
                role="user",
                content=f"I will query you 100 times through your API. You can only respond with either '{word1}' or '{word2}', and nothing else. Your task is to have {p}% of the answers be '{word1}', and the remaining {q}% should be '{word2}'. You can only respond with '{word1}' or '{word2}'. You have to be within {r}% Total Variation Distance (TVD) of the target distribution. Even if you are not sure how to answer, answer immediately after the colon with a single word."
            ),
            Message(role="assistant", content="My answer is: ")
        ]
        
        samples.append(Sample(
            prompt=prompt,
            target_distribution={word1: p/100, word2: q/100},
            word1=word1,
            word2=word2,
            tvd_tolerance=r/100,
            is_given_words=True,
            seed=43,  # Fixed seed for given words
            case_type="given_words"
        ))
    
    return samples


def get_combined_samples(num_examples: int = 10, shuffle: bool = True) -> List[Sample]:
    """Get combined samples from all three experiment types."""
    # Get samples from each type
    given_samples = get_given_words_samples(num_examples)
    random_samples = _generate_random_words_samples(num_examples, use_few_shot=False)
    icl_random_samples = _generate_random_words_samples(num_examples, use_few_shot=True)
    
    # Combine all samples
    all_samples = given_samples + random_samples + icl_random_samples
    
    # Shuffle if requested
    if shuffle:
        random.shuffle(all_samples)
    
    return all_samples

