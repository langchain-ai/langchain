from langchain_core.messages import (
    AIMessageChunk,
    add_ai_message_chunks,
)
from pydantic import SecretStr

from langchain_openai.chat_models.base import ChatOpenAI


def test_chat_openai_has_include_raw_response_field() -> None:
    """Test that ChatOpenAI has include_raw_response field."""

    model_true = ChatOpenAI(
        model="gpt-4", include_raw_response=True, api_key=SecretStr("fake-key")
    )
    assert model_true.include_raw_response is True

    model_false = ChatOpenAI(
        model="gpt-4", include_raw_response=False, api_key=SecretStr("fake-key")
    )
    assert model_false.include_raw_response is False

    model_default = ChatOpenAI(model="gpt-4", api_key=SecretStr("fake-key"))
    assert model_default.include_raw_response is False


def test_merge_streamed_chunks_preserves_raw_response() -> None:
    """Test that raw_response is preserved when merging chunks."""

    c1: AIMessageChunk = AIMessageChunk(
        content="Hel", raw_response={"choices": [{"delta": "Hel"}]}
    )
    c2: AIMessageChunk = AIMessageChunk(
        content="lo", raw_response={"choices": [{"delta": "lo"}]}
    )

    merged = add_ai_message_chunks(c1, c2)
    assert merged.content == "Hello"
    assert merged.raw_response == [
        {"choices": [{"delta": "Hel"}]},
        {"choices": [{"delta": "lo"}]},
    ]


def test_edge_case_empty_raw_response() -> None:
    """Test handling of chunks without raw_response."""

    c1: AIMessageChunk = AIMessageChunk(content="test", raw_response=None)
    merged = add_ai_message_chunks(c1)
    assert merged.raw_response is None


def test_single_chunk_raw_response_preserved() -> None:
    """Test that single chunk raw_response is preserved as dict."""

    c1: AIMessageChunk = AIMessageChunk(content="test", raw_response={"data": "value"})
    c2: AIMessageChunk = AIMessageChunk(content=" chunk", raw_response=None)

    merged = add_ai_message_chunks(c1, c2)
    # Single raw_response should be kept as dict, not list
    assert merged.raw_response == {"data": "value"}
