"""Test ChatAnthropic chat model."""

from typing import Optional

from langchain_core.messages import AIMessageChunk, BaseMessageChunk

from langchain_anthropic import ChatAnthropic, ChatAnthropicMessages

MODEL_NAME = "claude-3-5-haiku-latest"


def test_stream() -> None:
    """Test streaming tokens from Anthropic."""
    llm = ChatAnthropicMessages(model_name=MODEL_NAME)  # type: ignore[call-arg, call-arg]

    for token in llm.stream("I'm Pickle Rick"):
        pass


async def test_astream() -> None:
    """Test streaming tokens from Anthropic."""
    llm = ChatAnthropicMessages(model_name=MODEL_NAME)  # type: ignore[call-arg, call-arg]

    async for token in llm.astream("I'm Pickle Rick"):
        pass

    # Check expected raw API output
    async_client = llm._async_client
    params: dict = {
        "model": MODEL_NAME,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0,
    }
    stream = await async_client.messages.create(**params, stream=True)
    async for event in stream:
        pass


# async def test_stream_usage() -> None:
#     model = ChatAnthropic(model_name=MODEL_NAME, stream_usage=False)  # type: ignore[call-arg]
#     async for token in model.astream("hi"):
#         assert isinstance(token, AIMessageChunk)
#         assert token.usage_metadata is None

#     # check we override with kwarg
#     model = ChatAnthropic(model_name=MODEL_NAME)  # type: ignore[call-arg]
#     assert model.stream_usage
#     async for token in model.astream("hi", stream_usage=False):
#         assert isinstance(token, AIMessageChunk)
#         assert token.usage_metadata is None


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
