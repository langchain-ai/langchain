"""Test MiniMax chat model."""

import pytest  # type: ignore[import-not-found]

from langchain_minimax import ChatMiniMax

MODEL_NAME = "MiniMax-M2.5"


def test_initialization() -> None:
    """Test chat model initialization."""
    ChatMiniMax(model=MODEL_NAME)


def test_profile() -> None:
    """Test that model profile is loaded correctly."""
    model = ChatMiniMax(model="MiniMax-M2.5")
    assert model.profile


def test_minimax_model_param() -> None:
    """Test model name parameter handling."""
    llm = ChatMiniMax(model="foo")
    assert llm.model_name == "foo"
    llm = ChatMiniMax(model_name="foo")  # type: ignore[call-arg]
    assert llm.model_name == "foo"
    ls_params = llm._get_ls_params()
    assert ls_params.get("ls_provider") == "minimax"


def test_chat_minimax_extra_kwargs() -> None:
    """Test extra kwargs to chat minimax."""
    # Check that foo is saved in extra_kwargs.
    max_tokens = 10
    llm = ChatMiniMax(model=MODEL_NAME, foo=3, max_tokens=max_tokens)  # type: ignore[call-arg]
    assert llm.max_tokens == max_tokens
    assert llm.model_kwargs == {"foo": 3}

    # Test that if extra_kwargs are provided, they are added to it.
    llm = ChatMiniMax(model=MODEL_NAME, foo=3, model_kwargs={"bar": 2})  # type: ignore[call-arg]
    assert llm.model_kwargs == {"foo": 3, "bar": 2}

    # Test that if provided twice it errors
    with pytest.raises(ValueError, match="foo"):
        ChatMiniMax(model=MODEL_NAME, foo=3, model_kwargs={"foo": 2})  # type: ignore[call-arg]
