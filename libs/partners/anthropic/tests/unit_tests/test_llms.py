import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from anthropic.types import Message, TextBlock, ThinkingBlock, Usage

from langchain_anthropic import AnthropicLLM

os.environ["ANTHROPIC_API_KEY"] = "foo"


def test_anthropic_model_params() -> None:
    # Test standard tracing params
    with pytest.warns(
        expected_warning=DeprecationWarning,
        match=(
            "The class `AnthropicLLM` was deprecated in LangChain 0.1.0 "
            "and will be removed in 2.0.0. Use `ChatAnthropic` instead."
        ),
    ):
        llm = AnthropicLLM(model="foo")  # type: ignore[call-arg]

    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "anthropic",
        "ls_model_type": "llm",
        "ls_model_name": "foo",
        "ls_max_tokens": 1024,
    }

    llm = AnthropicLLM(model="foo", temperature=0.1)  # type: ignore[call-arg]

    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "anthropic",
        "ls_model_type": "llm",
        "ls_model_name": "foo",
        "ls_max_tokens": 1024,
        "ls_temperature": 0.1,
    }


def _thinking_first_message() -> Message:
    return Message(
        id="msg_1",
        type="message",
        role="assistant",
        model="claude-sonnet-5-5",
        content=[
            ThinkingBlock(type="thinking", thinking="", signature="sig"),
            TextBlock(type="text", text="foo"),
        ],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=Usage(input_tokens=1, output_tokens=1),
    )


def test_call_reads_text_blocks_by_type() -> None:
    """Responses that start with a thinking block still return their text."""
    llm = AnthropicLLM(model="claude-sonnet-5-5")  # type: ignore[call-arg]
    llm.client = MagicMock()
    llm.client.messages.create.return_value = _thinking_first_message()
    assert llm.invoke("Say foo:") == "foo"


async def test_acall_reads_text_blocks_by_type() -> None:
    """Async responses that start with a thinking block still return their text."""
    llm = AnthropicLLM(model="claude-sonnet-5-5")  # type: ignore[call-arg]
    llm.async_client = MagicMock()
    llm.async_client.messages.create = AsyncMock(return_value=_thinking_first_message())
    assert await llm.ainvoke("Say foo:") == "foo"
