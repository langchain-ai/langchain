"""Integration tests for ChatMiniMax."""

from __future__ import annotations

from langchain_minimax import ChatMiniMax

MODEL_NAME = "MiniMax-M2.5"


def test_invoke() -> None:
    """Test basic invoke."""
    chat_model = ChatMiniMax(model=MODEL_NAME)
    response = chat_model.invoke("Say hello in one word.")
    assert response.content


def test_streaming() -> None:
    """Test streaming."""
    chat_model = ChatMiniMax(model=MODEL_NAME)
    full = None
    for chunk in chat_model.stream("Say hello in one word."):
        full = chunk if full is None else full + chunk
    assert full is not None
    assert full.content
