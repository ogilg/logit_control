#!/usr/bin/env python3

"""
Minimal smoke tests for provider prompt formatting and dummy query helpers.

Usage:
  python test_format_prompt_harmony_lib.py

Requires the `openai-harmony` package (module name: `openai_harmony`).
Install with:
  pip install openai-harmony
"""

import importlib


def main() -> int:
    # Ensure Harmony lib is available before importing provider module
    try:
        import openai_harmony  # noqa: F401
    except Exception as e:
        print("openai_harmony is not installed. Install with: pip install openai-harmony")
        print(f"Import error: {e}")
        return 1

    import importlib
    import provider_wrapper.provider_wrapper.huggingface_provider as hf
    importlib.reload(hf)  # force fresh code


    from provider_wrapper.provider_wrapper.data_models import Message


    # Construct a minimal prompt with system and user
    prompt = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Say hi."),
    ]

    # GPT-OSS provider formatting (Harmony)
    gpt_provider = hf.GPTOSSProvider("gpt-oss-20b")
    gpt_tokens, gpt_text = gpt_provider.format_prompt(prompt)
    assert isinstance(gpt_tokens, list) and len(gpt_tokens) > 0
    assert isinstance(gpt_text, str)

    # Dummy get_text for GPT-OSS
    from provider_wrapper.provider_wrapper.data_models import GetTextRequest, GetProbsRequest

    txt_req = GetTextRequest(context=None, prompt=prompt, max_tokens=5, temperature=0.0)
    txt_resp_gpt = hf.dummy_get_text("gpt-oss-20b", txt_req)
    assert isinstance(txt_resp_gpt.txt, str) and len(txt_resp_gpt.txt) > 0
    assert txt_resp_gpt.context.get("harmony") is True

    # Dummy get_probs for GPT-OSS
    prob_req = GetProbsRequest(context=None, prompt=prompt, min_top_n=2, num_samples=None)
    prob_resp_gpt = hf.dummy_get_probs("gpt-oss-20b", prob_req)
    assert isinstance(prob_resp_gpt.probs, dict) and len(prob_resp_gpt.probs) == 2
    assert prob_resp_gpt.context.get("harmony") is True

    print("dummy + formatting tests OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


