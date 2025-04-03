import pytest

from langchain_openai import ChatOpenAI, OpenAI

_EXPECTED_NUM_TOKENS = {
    "ada": 17,
    "babbage": 17,
    "curie": 17,
    "davinci": 17,
    "gpt-4": 12,
    "gpt-4-32k": 12,
    "gpt-3.5-turbo": 12,
    "o1": 11,
    "o3": 11,
    "gpt-4o": 11,
}

_MODELS = models = ["ada", "babbage", "curie", "davinci"]
_CHAT_MODELS = ["gpt-4", "gpt-4-32k", "gpt-3.5-turbo", "o1", "o3", "gpt-4o"]


@pytest.mark.xfail(reason="Old models require different tiktoken cached file")
@pytest.mark.parametrize("model", _MODELS)
def test_openai_get_num_tokens(model: str) -> None:
    """Test get_tokens."""
    llm = OpenAI(model=model)
    assert llm.get_num_tokens("表情符号是\n🦜🔗") == _EXPECTED_NUM_TOKENS[model]


@pytest.mark.parametrize("model", _CHAT_MODELS)
def test_chat_openai_get_num_tokens(model: str) -> None:
    """Test get_tokens."""
    llm = ChatOpenAI(model=model)
    assert llm.get_num_tokens("表情符号是\n🦜🔗") == _EXPECTED_NUM_TOKENS[model]
