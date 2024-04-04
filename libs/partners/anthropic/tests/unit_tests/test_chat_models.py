"""Test chat model integration."""

import os

import pytest
from anthropic.types import ContentBlock, Message, Usage
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langchain_anthropic import ChatAnthropic, ChatAnthropicMessages
from langchain_anthropic.chat_models import _merge_messages

os.environ["ANTHROPIC_API_KEY"] = "foo"


def test_initialization() -> None:
    """Test chat model initialization."""
    ChatAnthropicMessages(model_name="claude-instant-1.2", anthropic_api_key="xyz")
    ChatAnthropicMessages(model="claude-instant-1.2", anthropic_api_key="xyz")


@pytest.mark.requires("anthropic")
def test_anthropic_model_name_param() -> None:
    llm = ChatAnthropic(model_name="foo")
    assert llm.model == "foo"


@pytest.mark.requires("anthropic")
def test_anthropic_model_param() -> None:
    llm = ChatAnthropic(model="foo")
    assert llm.model == "foo"


@pytest.mark.requires("anthropic")
def test_anthropic_model_kwargs() -> None:
    llm = ChatAnthropic(model_name="foo", model_kwargs={"foo": "bar"})
    assert llm.model_kwargs == {"foo": "bar"}


@pytest.mark.requires("anthropic")
def test_anthropic_invalid_model_kwargs() -> None:
    with pytest.raises(ValueError):
        ChatAnthropic(model="foo", model_kwargs={"max_tokens_to_sample": 5})


@pytest.mark.requires("anthropic")
def test_anthropic_incorrect_field() -> None:
    with pytest.warns(match="not default parameter"):
        llm = ChatAnthropic(model="foo", foo="bar")
    assert llm.model_kwargs == {"foo": "bar"}


@pytest.mark.requires("anthropic")
def test_anthropic_initialization() -> None:
    """Test anthropic initialization."""
    # Verify that chat anthropic can be initialized using a secret key provided
    # as a parameter rather than an environment variable.
    ChatAnthropic(model="test", anthropic_api_key="test")


def test__format_output() -> None:
    anthropic_msg = Message(
        id="foo",
        content=[ContentBlock(type="text", text="bar")],
        model="baz",
        role="assistant",
        stop_reason=None,
        stop_sequence=None,
        usage=Usage(input_tokens=2, output_tokens=1),
        type="message",
    )
    expected = ChatResult(
        generations=[
            ChatGeneration(message=AIMessage("bar")),
        ],
        llm_output={
            "id": "foo",
            "model": "baz",
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 2, "output_tokens": 1},
        },
    )
    llm = ChatAnthropic(model="test", anthropic_api_key="test")
    actual = llm._format_output(anthropic_msg)
    assert expected == actual


def test__merge_messages() -> None:
    messages = [
        SystemMessage("foo"),
        HumanMessage("bar"),
        AIMessage(
            [
                {"text": "baz", "type": "text"},
                {
                    "tool_input": {"a": "b"},
                    "type": "tool_use",
                    "id": "1",
                    "text": None,
                    "name": "buz",
                },
                {"text": "baz", "type": "text"},
                {
                    "tool_input": {"a": "c"},
                    "type": "tool_use",
                    "id": "2",
                    "text": None,
                    "name": "blah",
                },
            ]
        ),
        ToolMessage("buz output", tool_call_id="1"),
        ToolMessage("blah output", tool_call_id="2"),
        HumanMessage("next thing"),
    ]
    expected = [
        SystemMessage("foo"),
        HumanMessage("bar"),
        AIMessage(
            [
                {"text": "baz", "type": "text"},
                {
                    "tool_input": {"a": "b"},
                    "type": "tool_use",
                    "id": "1",
                    "text": None,
                    "name": "buz",
                },
                {"text": "baz", "type": "text"},
                {
                    "tool_input": {"a": "c"},
                    "type": "tool_use",
                    "id": "2",
                    "text": None,
                    "name": "blah",
                },
            ]
        ),
        HumanMessage(
            [
                {"type": "tool_result", "content": "buz output", "tool_use_id": "1"},
                {"type": "tool_result", "content": "blah output", "tool_use_id": "2"},
                {"type": "text", "text": "next thing"},
            ]
        ),
    ]
    actual = _merge_messages(messages)
    assert expected == actual
