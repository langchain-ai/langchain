import pytest

from langchain_neospace import ChatNeoSpace, NeoSpace

_EXPECTED_NUM_TOKENS = {
    "ada": 17,
    "babbage": 17,
    "curie": 17,
    "davinci": 17,
    "neo-4": 12,
    "neo-4-32k": 12,
    "neo-3.5-turbo": 12,
}

_MODELS = models = ["ada", "babbage", "curie", "davinci"]
_CHAT_MODELS = ["neo-4", "neo-4-32k", "neo-3.5-turbo"]


@pytest.mark.parametrize("model", _MODELS)
def test_neospace_get_num_tokens(model: str) -> None:
    """Test get_tokens."""
    llm = NeoSpace(model=model)
    assert llm.get_num_tokens("表情符号是\n🦜🔗") == _EXPECTED_NUM_TOKENS[model]


@pytest.mark.parametrize("model", _CHAT_MODELS)
def test_chat_neospace_get_num_tokens(model: str) -> None:
    """Test get_tokens."""
    llm = ChatNeoSpace(model=model)
    assert llm.get_num_tokens("表情符号是\n🦜🔗") == _EXPECTED_NUM_TOKENS[model]
