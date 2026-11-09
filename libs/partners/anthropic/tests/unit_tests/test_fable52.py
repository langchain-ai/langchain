from typing import Any, cast

import pytest
from langchain_core.runnables import RunnableBinding

from langchain_anthropic import ChatAnthropic


@pytest.mark.parametrize("model", ["claude-fable-5-2", "claude-mythos-5-2"])
def test_profiles(model: str) -> None:
    profile = ChatAnthropic(model=model).profile
    assert profile is not None
    if model == "claude-fable-5-2":
        assert profile["max_input_tokens"] == 1_000_000
        assert profile["max_output_tokens"] == 128_000
    assert profile["structured_output"] is True
    assert profile["tool_choice"] is False
    assert profile["mid_conversation_system_messages"] is True
    assert profile["mid_conversation_tools"] is True
    assert profile["reasoning_effort_default"] == "high"


@pytest.mark.parametrize("model", ["claude-fable-5-2", "claude-mythos-5-2"])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"thinking": {"type": "disabled"}},
        {"thinking": {"type": "enabled", "budget_tokens": 1024}},
        {"temperature": 0.1},
        {"top_p": 0.9},
        {"top_k": 1},
    ],
)
def test_unsupported_configuration(model: str, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="not supported"):
        ChatAnthropic(model=model, **kwargs)._get_request_payload("Hello")


@pytest.mark.parametrize("model", ["claude-fable-5-2", "claude-mythos-5-2"])
def test_request(model: str) -> None:
    llm = ChatAnthropic(model=model, reasoning_effort="xhigh")
    payload = llm._get_request_payload("Hello", speed="fast")
    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert payload["output_config"] == {"effort": "xhigh"}
    assert payload["speed"] == "fast"
    assert not {"temperature", "top_p", "top_k"} & payload.keys()
    toolset = {"type": "computer_toolset_20260801"}
    assert cast("RunnableBinding", llm.bind_tools([toolset])).kwargs["tools"] == [
        toolset
    ]


def test_sdk_upgrade_preserves_api_version() -> None:
    model = ChatAnthropic(model="claude-fable-5-2", api_key="test")
    assert model._client.default_headers["anthropic-version"] == "2023-06-01"
    override = ChatAnthropic(
        model="claude-fable-5-2",
        api_key="test",
        default_headers={"anthropic-version": "2026-10-21"},
    )
    assert override._client.default_headers["anthropic-version"] == "2026-10-21"


@pytest.mark.parametrize("model", ["claude-fable-5-2", "claude-mythos-5-2"])
def test_implicit_thinking_stream(model: str) -> None:
    from unittest.mock import Mock, patch

    from anthropic.types import RawContentBlockStartEvent, ThinkingBlock
    from langchain_core.messages import HumanMessage

    llm = ChatAnthropic(model=model)
    event = RawContentBlockStartEvent(
        type="content_block_start",
        index=0,
        content_block=ThinkingBlock(type="thinking", thinking="", signature="signed"),
    )
    with patch.object(
        ChatAnthropic, "_create", return_value=Mock(parse=lambda: [event])
    ):
        chunks = list(llm._stream([HumanMessage("Hello")]))
    assert chunks[0].message.content == [
        {"type": "thinking", "thinking": "", "signature": "signed", "index": 0}
    ]
