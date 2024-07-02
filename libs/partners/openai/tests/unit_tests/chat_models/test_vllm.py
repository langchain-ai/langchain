"""Test Azure OpenAI Chat API wrapper."""

from langchain_openai import VLLMChatOpenAI


def test_initialize_vllm_openai() -> None:
    llm = VLLMChatOpenAI(
        model_name="foo", openai_api_key="xyz", openai_api_base="https://test.com"
    )
    assert llm.model_name == "foo"
    assert llm.openai_api_base == "https://test.com"


def test_initialize_more() -> None:
    llm = VLLMChatOpenAI(
        model_name="foo",
        openai_api_key="xyz",
        openai_api_base="https://test.com",
        temperature=0,
        best_of=1,
        use_beam_search=False,
        top_k=100,
        min_p=0,
        repetition_penalty=2,
        length_penalty=1,
        early_stopping=True,
        stop_token_ids=[1, 2, 3],
        ignore_eos=True,
        min_tokens=10,
        add_generation_prompt=False,
        add_special_tokens=True,
        guided_regex=r"[A-Z]+",
    )
    assert llm.openai_api_key is not None
    assert llm.openai_api_key.get_secret_value() == "xyz"
    assert llm.openai_api_base == "https://test.com"
    assert llm.temperature == 0
    assert llm.best_of == 1
    assert llm.use_beam_search is False
    assert llm.top_k == 100
    assert llm.min_p == 0
    assert llm.repetition_penalty == 2
    assert llm.length_penalty == 1
    assert llm.early_stopping is True
    assert llm.stop_token_ids == [1, 2, 3]
    assert llm.ignore_eos is True
    assert llm.min_tokens == 10
    assert llm.add_generation_prompt is False
    assert llm.add_special_tokens is True
    assert llm.guided_regex == "[A-Z]+"
