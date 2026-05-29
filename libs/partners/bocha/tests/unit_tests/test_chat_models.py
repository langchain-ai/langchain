"""Unit tests for ChatBocha chat model."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import BaseMessage, HumanMessage

from langchain_bocha._client import BochaClient
from langchain_bocha.chat_models import ChatBocha

MOCK_CHAT_RESPONSE = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "Hello! I am a helpful AI assistant.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
    },
}

STREAM_CHUNKS = [
    {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
    {"choices": [{"delta": {"content": "!"}, "finish_reason": None}]},
]


def _make_model() -> ChatBocha:
    """Create a ChatBocha instance with a fake client."""
    return ChatBocha(api_key="fake-key")  # type: ignore[arg-type]


def test_chat_bocha_initialization() -> None:
    """Test ChatBocha initialization and validation."""
    model = ChatBocha(
        model="deepseek-v4-pro",
        api_key="fake-key",  # type: ignore[arg-type]
    )
    assert model.model == "deepseek-v4-pro"
    assert model._llm_type == "chat-bocha"
    assert model.api_base == "https://api.bocha.cn/v1"
    assert isinstance(model.client, BochaClient)


def test_chat_bocha_validate_environment_missing_key() -> None:
    """Test environment validation raises error when API key is missing."""
    with pytest.raises(ValueError, match="BOCHA_API_KEY must be set"):
        ChatBocha(
            model="deepseek-v4-pro",
            api_key=None,
        )


def test_chat_bocha_invoke() -> None:
    """Test ChatBocha invoke with mocked client."""
    model = _make_model()
    model.client = MagicMock(spec=BochaClient)
    model.client.post.return_value = MOCK_CHAT_RESPONSE

    messages = [HumanMessage(content="Hello")]
    response = model.invoke(messages)

    assert response.content == "Hello! I am a helpful AI assistant."
    assert "choices" in response.response_metadata
    assert response.usage_metadata is not None
    assert response.usage_metadata["input_tokens"] == 10
    assert response.usage_metadata["output_tokens"] == 20
    assert response.usage_metadata["total_tokens"] == 30
    model.client.post.assert_called_once()


@pytest.mark.asyncio
async def test_chat_bocha_ainvoke() -> None:
    """Test ChatBocha async invoke with mocked client."""
    model = _make_model()
    mock_client = MagicMock(spec=BochaClient)
    from unittest.mock import AsyncMock

    mock_client.apost = AsyncMock(return_value=MOCK_CHAT_RESPONSE)
    model.client = mock_client

    messages = [HumanMessage(content="Hello")]
    response = await model.ainvoke(messages)

    assert response.content == "Hello! I am a helpful AI assistant."
    mock_client.apost.assert_called_once()


def test_chat_bocha_stream() -> None:
    """Test ChatBocha stream with mocked client."""
    model = _make_model()
    model.client = MagicMock(spec=BochaClient)
    model.client.post_stream.return_value = iter(STREAM_CHUNKS)

    messages = [HumanMessage(content="Hello")]
    chunks = list(model.stream(messages))

    assert len(chunks) >= 2
    content = "".join(str(c.content) for c in chunks)
    assert content == "Hello!"
    assert chunks[0].response_metadata["choices"][0]["delta"]["content"] == "Hello"
    model.client.post_stream.assert_called_once()


@pytest.mark.asyncio
async def test_chat_bocha_astream() -> None:
    """Test ChatBocha async stream with mocked client."""
    model = _make_model()
    mock_client = MagicMock(spec=BochaClient)

    async def mock_apost_stream(*args, **kwargs):  # type: ignore[no-untyped-def]
        for chunk in STREAM_CHUNKS:
            yield chunk

    mock_client.apost_stream = mock_apost_stream
    model.client = mock_client

    messages = [HumanMessage(content="Hello")]
    chunks = [chunk async for chunk in model.astream(messages)]

    assert len(chunks) >= 2
    content = "".join(str(c.content) for c in chunks)
    assert content == "Hello!"
    assert chunks[0].response_metadata["choices"][0]["delta"]["content"] == "Hello"


def test_chat_bocha_streaming_flag() -> None:
    """Test that streaming=True delegates _generate to _stream."""
    model = ChatBocha(api_key="fake-key", streaming=True)  # type: ignore[arg-type]
    model.client = MagicMock(spec=BochaClient)
    model.client.post_stream.return_value = iter(STREAM_CHUNKS)

    messages: list[BaseMessage] = [HumanMessage(content="Hello")]
    result = model._generate(messages)

    assert result.generations[0].message.content == "Hello!"
    model.client.post_stream.assert_called_once()
