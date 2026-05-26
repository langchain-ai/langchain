"""Integration tests for `ChatAtlas`."""

from __future__ import annotations

from langchain_core.messages import AIMessageChunk, BaseMessageChunk

from langchain_atlas.chat_models import ChatAtlas

MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"


def test_basic_invoke() -> None:
    """Test basic Atlas invocation."""
    chat_model = ChatAtlas(model=MODEL_NAME, temperature=0)
    response = chat_model.invoke("Reply with 'atlas ok' and nothing else.")
    assert response.content


def test_streaming() -> None:
    """Test Atlas streaming responses."""
    chat_model = ChatAtlas(model=MODEL_NAME, temperature=0)
    full: BaseMessageChunk | None = None
    for chunk in chat_model.stream("Reply with 'stream ok' and nothing else."):
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert full.content
