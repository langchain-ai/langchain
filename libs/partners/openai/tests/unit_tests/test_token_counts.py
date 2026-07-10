import pytest

from langchain_openai import ChatOpenAI, OpenAI

_EXPECTED_NUM_TOKENS = {
    "ada": 17,
    "babbage": 17,
    "curie": 17,
    "davinci": 17,
    "gpt-5.5": 11,
    "gpt-5-nano": 11,
    "o1": 11,
    "o1-preview": 11,
    "o1-mini": 11,
    "o3": 11,
    "o3-mini": 11,
    "o3-pro": 11,
}

_MODELS = models = ["ada", "babbage", "curie", "davinci"]
_CHAT_MODELS = ["gpt-5.5", "gpt-5-nano", "o1", "o1-preview", "o1-mini", "o3", "o3-mini", "o3-pro"]


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
