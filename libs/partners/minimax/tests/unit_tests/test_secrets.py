"""Test secret masking for MiniMax chat model."""

from langchain_minimax import ChatMiniMax

MODEL_NAME = "MiniMax-M2.5"


def test_chat_minimax_secrets() -> None:
    """Test that API key is not exposed in string representation."""
    o = ChatMiniMax(model=MODEL_NAME, minimax_api_key="foo")  # type: ignore[call-arg]
    s = str(o)
    assert "foo" not in s
