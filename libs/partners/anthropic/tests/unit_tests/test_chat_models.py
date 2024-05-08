"""Test chat model integration."""

import os
from typing import Any, Callable, Dict, Literal, Type, cast

import pytest
from anthropic.types import ContentBlock, Message, Usage
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr
from langchain_core.tools import BaseTool
from pytest import CaptureFixture, MonkeyPatch

from langchain_anthropic import ChatAnthropic
from langchain_anthropic.chat_models import (
    _format_messages,
    _merge_messages,
    convert_to_anthropic_tool,
)

os.environ["ANTHROPIC_API_KEY"] = "foo"


def test_initialization() -> None:
    """Test chat model initialization."""
    for model in [
        ChatAnthropic(model_name="claude-instant-1.2", api_key="xyz", timeout=2),
        ChatAnthropic(
            model="claude-instant-1.2",
            anthropic_api_key="xyz",
            default_request_timeout=2,
        ),
    ]:
        assert model.model == "claude-instant-1.2"
        assert cast(SecretStr, model.anthropic_api_key).get_secret_value() == "xyz"
        assert model.default_request_timeout == 2.0


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


def test__merge_messages_mutation() -> None:
    original_messages = [
        HumanMessage([{"type": "text", "text": "bar"}]),
        HumanMessage("next thing"),
    ]
    messages = [
        HumanMessage([{"type": "text", "text": "bar"}]),
        HumanMessage("next thing"),
    ]
    expected = [
        HumanMessage(
            [{"type": "text", "text": "bar"}, {"type": "text", "text": "next thing"}]
        ),
    ]
    actual = _merge_messages(messages)
    assert expected == actual
    assert messages == original_messages


@pytest.fixture()
def pydantic() -> Type[BaseModel]:
    class dummy_function(BaseModel):
        """dummy function"""

        arg1: int = Field(..., description="foo")
        arg2: Literal["bar", "baz"] = Field(..., description="one of 'bar', 'baz'")

    return dummy_function


@pytest.fixture()
def function() -> Callable:
    def dummy_function(arg1: int, arg2: Literal["bar", "baz"]) -> None:
        """dummy function

        Args:
            arg1: foo
            arg2: one of 'bar', 'baz'
        """
        pass

    return dummy_function


@pytest.fixture()
def dummy_tool() -> BaseTool:
    class Schema(BaseModel):
        arg1: int = Field(..., description="foo")
        arg2: Literal["bar", "baz"] = Field(..., description="one of 'bar', 'baz'")

    class DummyFunction(BaseTool):
        args_schema: Type[BaseModel] = Schema
        name: str = "dummy_function"
        description: str = "dummy function"

        def _run(self, *args: Any, **kwargs: Any) -> Any:
            pass

    return DummyFunction()


@pytest.fixture()
def json_schema() -> Dict:
    return {
        "title": "dummy_function",
        "description": "dummy function",
        "type": "object",
        "properties": {
            "arg1": {"description": "foo", "type": "integer"},
            "arg2": {
                "description": "one of 'bar', 'baz'",
                "enum": ["bar", "baz"],
                "type": "string",
            },
        },
        "required": ["arg1", "arg2"],
    }


@pytest.fixture()
def openai_function() -> Dict:
    return {
        "name": "dummy_function",
        "description": "dummy function",
        "parameters": {
            "type": "object",
            "properties": {
                "arg1": {"description": "foo", "type": "integer"},
                "arg2": {
                    "description": "one of 'bar', 'baz'",
                    "enum": ["bar", "baz"],
                    "type": "string",
                },
            },
            "required": ["arg1", "arg2"],
        },
    }


def test_convert_to_anthropic_tool(
    pydantic: Type[BaseModel],
    function: Callable,
    dummy_tool: BaseTool,
    json_schema: Dict,
    openai_function: Dict,
) -> None:
    expected = {
        "name": "dummy_function",
        "description": "dummy function",
        "input_schema": {
            "type": "object",
            "properties": {
                "arg1": {"description": "foo", "type": "integer"},
                "arg2": {
                    "description": "one of 'bar', 'baz'",
                    "enum": ["bar", "baz"],
                    "type": "string",
                },
            },
            "required": ["arg1", "arg2"],
        },
    }

    for fn in (pydantic, function, dummy_tool, json_schema, expected, openai_function):
        actual = convert_to_anthropic_tool(fn)  # type: ignore
        assert actual == expected


def test__format_messages_with_tool_calls() -> None:
    system = SystemMessage("fuzz")
    human = HumanMessage("foo")
    ai = AIMessage(
        "",
        tool_calls=[{"name": "bar", "id": "1", "args": {"baz": "buzz"}}],
    )
    tool = ToolMessage(
        "blurb",
        tool_call_id="1",
    )
    messages = [system, human, ai, tool]
    expected = (
        "fuzz",
        [
            {"role": "user", "content": "foo"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "bar",
                        "id": "1",
                        "input": {"baz": "buzz"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "content": "blurb", "tool_use_id": "1"}
                ],
            },
        ],
    )
    actual = _format_messages(messages)
    assert expected == actual


def test__format_messages_with_str_content_and_tool_calls() -> None:
    system = SystemMessage("fuzz")
    human = HumanMessage("foo")
    # If content and tool_calls are specified and content is a string, then both are
    # included with content first.
    ai = AIMessage(
        "thought",
        tool_calls=[{"name": "bar", "id": "1", "args": {"baz": "buzz"}}],
    )
    tool = ToolMessage(
        "blurb",
        tool_call_id="1",
    )
    messages = [system, human, ai, tool]
    expected = (
        "fuzz",
        [
            {"role": "user", "content": "foo"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "thought",
                    },
                    {
                        "type": "tool_use",
                        "name": "bar",
                        "id": "1",
                        "input": {"baz": "buzz"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "content": "blurb", "tool_use_id": "1"}
                ],
            },
        ],
    )
    actual = _format_messages(messages)
    assert expected == actual


def test__format_messages_with_list_content_and_tool_calls() -> None:
    system = SystemMessage("fuzz")
    human = HumanMessage("foo")
    # If content and tool_calls are specified and content is a list, then content is
    # preferred.
    ai = AIMessage(
        [
            {
                "type": "text",
                "text": "thought",
            }
        ],
        tool_calls=[{"name": "bar", "id": "1", "args": {"baz": "buzz"}}],
    )
    tool = ToolMessage(
        "blurb",
        tool_call_id="1",
    )
    messages = [system, human, ai, tool]
    expected = (
        "fuzz",
        [
            {"role": "user", "content": "foo"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "thought",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "content": "blurb", "tool_use_id": "1"}
                ],
            },
        ],
    )
    actual = _format_messages(messages)
    assert expected == actual


def test_anthropic_api_key_is_secret_string() -> None:
    """Test that the API key is stored as a SecretStr."""
    chat_model = ChatAnthropic(
        model="claude-3-opus-20240229",
        anthropic_api_key="secret-api-key",
    )
    assert isinstance(chat_model.anthropic_api_key, SecretStr)


def test_anthropic_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test that the API key is masked when passed from an environment variable."""
    monkeypatch.setenv("ANTHROPIC_API_KEY ", "secret-api-key")
    chat_model = ChatAnthropic(
        model="claude-3-opus-20240229",
    )
    print(chat_model.anthropic_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_anthropic_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test that the API key is masked when passed via the constructor."""
    chat_model = ChatAnthropic(
        model="claude-3-opus-20240229",
        anthropic_api_key="secret-api-key",
    )
    print(chat_model.anthropic_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_anthropic_uses_actual_secret_value_from_secretstr() -> None:
    """Test that the actual secret value is correctly retrieved."""
    chat_model = ChatAnthropic(
        model="claude-3-opus-20240229",
        anthropic_api_key="secret-api-key",
    )
    assert (
        cast(SecretStr, chat_model.anthropic_api_key).get_secret_value()
        == "secret-api-key"
    )
