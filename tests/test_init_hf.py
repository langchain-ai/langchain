import sys, os
sys.path.append(os.path.abspath("libs/langchain"))  # ensures local package can be imported

from libs.langchain.langchain_classic.chat_models import init_chat_model


def test_hf_model_init():
    llm = init_chat_model(
        model="microsoft/Phi-3-mini-4k-instruct",
        model_provider="huggingface",
        temperature=0,
        max_tokens=1024,
        timeout=None,
        max_retries=2,
    )
    print(llm)  # optional for debug
    assert llm is not None
