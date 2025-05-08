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

    full: Optional[BaseMessageChunk] = None
    chunks_with_input_token_counts = 0
    chunks_with_output_token_counts = 0
    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)
        full = token if full is None else full + token
        assert isinstance(token, AIMessageChunk)
        if token.usage_metadata is not None:
            if token.usage_metadata.get("input_tokens"):
                chunks_with_input_token_counts += 1
            if token.usage_metadata.get("output_tokens"):
                chunks_with_output_token_counts += 1
    if chunks_with_input_token_counts != 1 or chunks_with_output_token_counts != 1:
        raise AssertionError(
            "Expected exactly one chunk with input or output token counts. "
            "AIMessageChunk aggregation adds counts. Check that "
            "this is behaving properly."
        )
    # check token usage is populated
    assert isinstance(full, AIMessageChunk)
    assert full.usage_metadata is not None
    assert full.usage_metadata["input_tokens"] > 0
    assert full.usage_metadata["output_tokens"] > 0
    assert full.usage_metadata["total_tokens"] > 0
    assert (
        full.usage_metadata["input_tokens"] + full.usage_metadata["output_tokens"]
        == full.usage_metadata["total_tokens"]
    )
    assert "stop_reason" in full.response_metadata
    assert "stop_sequence" in full.response_metadata

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
        if event.type == "message_start":
            assert event.message.usage.input_tokens > 1
            # Note: this single output token included in message start event
            # does not appear to contribute to overall output token counts. It
            # is excluded from the total token count.
            assert event.message.usage.output_tokens == 1
        elif event.type == "message_delta":
            assert event.usage.output_tokens > 1
        else:
            pass


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
