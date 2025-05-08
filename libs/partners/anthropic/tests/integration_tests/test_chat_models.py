"""Test ChatAnthropic chat model."""

from langchain_core.messages import AIMessageChunk

from langchain_anthropic import ChatAnthropic

MODEL_NAME = "claude-3-5-haiku-latest"

async def test_stream_usage() -> None:
    model = ChatAnthropic(model_name=MODEL_NAME, stream_usage=False)  # type: ignore[call-arg]
    async for token in model.astream("hi"):
        assert isinstance(token, AIMessageChunk)
        assert token.usage_metadata is None

    # check we override with kwarg
    model = ChatAnthropic(model_name=MODEL_NAME)  # type: ignore[call-arg]
    assert model.stream_usage
    async for token in model.astream("hi", stream_usage=False):
        assert isinstance(token, AIMessageChunk)
        assert token.usage_metadata is None


async def test_async_stream_twice() -> None:
    model = ChatAnthropic(model_name=MODEL_NAME, stream_usage=False)  # type: ignore[call-arg]
    async for token in model.astream("hi"):
        assert isinstance(token, AIMessageChunk)
        assert token.usage_metadata is None

    # check we override with kwarg
    model = ChatAnthropic(model_name=MODEL_NAME)  # type: ignore[call-arg]
    assert model.stream_usage
    async for token in model.astream("hi", stream_usage=False):
        assert isinstance(token, AIMessageChunk)
        assert token.usage_metadata is None
