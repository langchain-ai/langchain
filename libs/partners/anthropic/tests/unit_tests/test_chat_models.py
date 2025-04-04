"""Test chat model integration."""

import os
from typing import Any, Callable, Dict, Literal, Type, cast

import pytest
from anthropic.types import Message, TextBlock, Usage
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableBinding
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, SecretStr
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
        ChatAnthropic(model_name="claude-instant-1.2", api_key="xyz", timeout=2),  # type: ignore[arg-type, call-arg]
        ChatAnthropic(  # type: ignore[call-arg, call-arg, call-arg]
            model="claude-instant-1.2",
            anthropic_api_key="xyz",
            default_request_timeout=2,
            base_url="https://api.anthropic.com",
        ),
    ]:
        assert model.model == "claude-instant-1.2"
        assert cast(SecretStr, model.anthropic_api_key).get_secret_value() == "xyz"
        assert model.default_request_timeout == 2.0
        assert model.anthropic_api_url == "https://api.anthropic.com"


@pytest.mark.requires("anthropic")
def test_anthropic_model_name_param() -> None:
    llm = ChatAnthropic(model_name="foo")  # type: ignore[call-arg, call-arg]
    assert llm.model == "foo"


@pytest.mark.requires("anthropic")
def test_anthropic_model_param() -> None:
    llm = ChatAnthropic(model="foo")  # type: ignore[call-arg]
    assert llm.model == "foo"


@pytest.mark.requires("anthropic")
def test_anthropic_model_kwargs() -> None:
    llm = ChatAnthropic(model_name="foo", model_kwargs={"foo": "bar"})  # type: ignore[call-arg, call-arg]
    assert llm.model_kwargs == {"foo": "bar"}


@pytest.mark.requires("anthropic")
def test_anthropic_fields_in_model_kwargs() -> None:
    """Test that for backwards compatibility fields can be passed in as model_kwargs."""
    llm = ChatAnthropic(model="foo", model_kwargs={"max_tokens_to_sample": 5})  # type: ignore[call-arg]
    assert llm.max_tokens == 5
    llm = ChatAnthropic(model="foo", model_kwargs={"max_tokens": 5})  # type: ignore[call-arg]
    assert llm.max_tokens == 5


@pytest.mark.requires("anthropic")
def test_anthropic_incorrect_field() -> None:
    with pytest.warns(match="not default parameter"):
        llm = ChatAnthropic(model="foo", foo="bar")  # type: ignore[call-arg, call-arg]
    assert llm.model_kwargs == {"foo": "bar"}


@pytest.mark.requires("anthropic")
def test_anthropic_initialization() -> None:
    """Test anthropic initialization."""
    # Verify that chat anthropic can be initialized using a secret key provided
    # as a parameter rather than an environment variable.
    ChatAnthropic(model="test", anthropic_api_key="test")  # type: ignore[call-arg, call-arg]


def test__format_output() -> None:
    anthropic_msg = Message(
        id="foo",
        content=[TextBlock(type="text", text="bar")],
        model="baz",
        role="assistant",
        stop_reason=None,
        stop_sequence=None,
        usage=Usage(input_tokens=2, output_tokens=1),
        type="message",
    )
    expected = AIMessage(  # type: ignore[misc]
        "bar",
        usage_metadata={
            "input_tokens": 2,
            "output_tokens": 1,
            "total_tokens": 3,
            "input_token_details": {},
        },
    )
    llm = ChatAnthropic(model="test", anthropic_api_key="test")  # type: ignore[call-arg, call-arg]
    actual = llm._format_output(anthropic_msg)
    assert actual.generations[0].message == expected


def test__format_output_cached() -> None:
    anthropic_msg = Message(
        id="foo",
        content=[TextBlock(type="text", text="bar")],
        model="baz",
        role="assistant",
        stop_reason=None,
        stop_sequence=None,
        usage=Usage(
            input_tokens=2,
            output_tokens=1,
            cache_creation_input_tokens=3,
            cache_read_input_tokens=4,
        ),
        type="message",
    )
    expected = AIMessage(  # type: ignore[misc]
        "bar",
        usage_metadata={
            "input_tokens": 9,
            "output_tokens": 1,
            "total_tokens": 10,
            "input_token_details": {"cache_creation": 3, "cache_read": 4},
        },
    )

    llm = ChatAnthropic(model="test", anthropic_api_key="test")  # type: ignore[call-arg, call-arg]
    actual = llm._format_output(anthropic_msg)
    assert actual.generations[0].message == expected


def test__merge_messages() -> None:
    messages = [
        SystemMessage("foo"),  # type: ignore[misc]
        HumanMessage("bar"),  # type: ignore[misc]
        AIMessage(  # type: ignore[misc]
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
                {
                    "tool_input": {"a": "c"},
                    "type": "tool_use",
                    "id": "3",
                    "text": None,
                    "name": "blah",
                },
            ]
        ),
        ToolMessage("buz output", tool_call_id="1", status="error"),  # type: ignore[misc]
        ToolMessage(
            content=[
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "fake_image_data",
                    },
                },
            ],
            tool_call_id="2",
        ),  # type: ignore[misc]
        ToolMessage([], tool_call_id="3"),  # type: ignore[misc]
        HumanMessage("next thing"),  # type: ignore[misc]
    ]
    expected = [
        SystemMessage("foo"),  # type: ignore[misc]
        HumanMessage("bar"),  # type: ignore[misc]
        AIMessage(  # type: ignore[misc]
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
                {
                    "tool_input": {"a": "c"},
                    "type": "tool_use",
                    "id": "3",
                    "text": None,
                    "name": "blah",
                },
            ]
        ),
        HumanMessage(  # type: ignore[misc]
            [
                {
                    "type": "tool_result",
                    "content": "buz output",
                    "tool_use_id": "1",
                    "is_error": True,
                },
                {
                    "type": "tool_result",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "fake_image_data",
                            },
                        },
                    ],
                    "tool_use_id": "2",
                    "is_error": False,
                },
                {
                    "type": "tool_result",
                    "content": [],
                    "tool_use_id": "3",
                    "is_error": False,
                },
                {"type": "text", "text": "next thing"},
            ]
        ),
    ]
    actual = _merge_messages(messages)
    assert expected == actual

    # Test tool message case
    messages = [
        ToolMessage("buz output", tool_call_id="1"),  # type: ignore[misc]
        ToolMessage(  # type: ignore[misc]
            content=[
                {"type": "tool_result", "content": "blah output", "tool_use_id": "2"}
            ],
            tool_call_id="2",
        ),
    ]
    expected = [
        HumanMessage(  # type: ignore[misc]
            [
                {
                    "type": "tool_result",
                    "content": "buz output",
                    "tool_use_id": "1",
                    "is_error": False,
                },
                {"type": "tool_result", "content": "blah output", "tool_use_id": "2"},
            ]
        )
    ]
    actual = _merge_messages(messages)
    assert expected == actual


def test__merge_messages_mutation() -> None:
    original_messages = [
        HumanMessage([{"type": "text", "text": "bar"}]),  # type: ignore[misc]
        HumanMessage("next thing"),  # type: ignore[misc]
    ]
    messages = [
        HumanMessage([{"type": "text", "text": "bar"}]),  # type: ignore[misc]
        HumanMessage("next thing"),  # type: ignore[misc]
    ]
    expected = [
        HumanMessage(  # type: ignore[misc]
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

    class DummyFunction(BaseTool):  # type: ignore[override]
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
    system = SystemMessage("fuzz")  # type: ignore[misc]
    human = HumanMessage("foo")  # type: ignore[misc]
    ai = AIMessage(
        "",  # with empty string
        tool_calls=[{"name": "bar", "id": "1", "args": {"baz": "buzz"}}],
    )
    ai2 = AIMessage(
        [],  # with empty list
        tool_calls=[{"name": "bar", "id": "2", "args": {"baz": "buzz"}}],
    )
    tool = ToolMessage(
        "blurb",
        tool_call_id="1",
    )
    tool_image_url = ToolMessage(
        [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...."}}],
        tool_call_id="2",
    )
    tool_image = ToolMessage(
        [
            {
                "type": "image",
                "source": {
                    "data": "....",
                    "type": "base64",
                    "media_type": "image/jpeg",
                },
            }
        ],
        tool_call_id="3",
    )
    messages = [system, human, ai, tool, ai2, tool_image_url, tool_image]
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
                    {
                        "type": "tool_result",
                        "content": "blurb",
                        "tool_use_id": "1",
                        "is_error": False,
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "bar",
                        "id": "2",
                        "input": {"baz": "buzz"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "data": "....",
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                },
                            }
                        ],
                        "tool_use_id": "2",
                        "is_error": False,
                    },
                    {
                        "type": "tool_result",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "data": "....",
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                },
                            }
                        ],
                        "tool_use_id": "3",
                        "is_error": False,
                    },
                ],
            },
        ],
    )
    actual = _format_messages(messages)
    assert expected == actual


def test__format_messages_with_str_content_and_tool_calls() -> None:
    system = SystemMessage("fuzz")  # type: ignore[misc]
    human = HumanMessage("foo")  # type: ignore[misc]
    # If content and tool_calls are specified and content is a string, then both are
    # included with content first.
    ai = AIMessage(  # type: ignore[misc]
        "thought",
        tool_calls=[{"name": "bar", "id": "1", "args": {"baz": "buzz"}}],
    )
    tool = ToolMessage("blurb", tool_call_id="1")  # type: ignore[misc]
    messages = [system, human, ai, tool]
    expected = (
        "fuzz",
        [
            {"role": "user", "content": "foo"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "thought"},
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
                    {
                        "type": "tool_result",
                        "content": "blurb",
                        "tool_use_id": "1",
                        "is_error": False,
                    }
                ],
            },
        ],
    )
    actual = _format_messages(messages)
    assert expected == actual


def test__format_messages_with_list_content_and_tool_calls() -> None:
    system = SystemMessage("fuzz")  # type: ignore[misc]
    human = HumanMessage("foo")  # type: ignore[misc]
    ai = AIMessage(  # type: ignore[misc]
        [{"type": "text", "text": "thought"}],
        tool_calls=[{"name": "bar", "id": "1", "args": {"baz": "buzz"}}],
    )
    tool = ToolMessage(  # type: ignore[misc]
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
                    {"type": "text", "text": "thought"},
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
                    {
                        "type": "tool_result",
                        "content": "blurb",
                        "tool_use_id": "1",
                        "is_error": False,
                    }
                ],
            },
        ],
    )
    actual = _format_messages(messages)
    assert expected == actual


def test__format_messages_with_tool_use_blocks_and_tool_calls() -> None:
    """Show that tool_calls are preferred to tool_use blocks when both have same id."""
    system = SystemMessage("fuzz")  # type: ignore[misc]
    human = HumanMessage("foo")  # type: ignore[misc]
    # NOTE: tool_use block in contents and tool_calls have different arguments.
    ai = AIMessage(  # type: ignore[misc]
        [
            {"type": "text", "text": "thought"},
            {
                "type": "tool_use",
                "name": "bar",
                "id": "1",
                "input": {"baz": "NOT_BUZZ"},
            },
        ],
        tool_calls=[{"name": "bar", "id": "1", "args": {"baz": "BUZZ"}}],
    )
    tool = ToolMessage("blurb", tool_call_id="1")  # type: ignore[misc]
    messages = [system, human, ai, tool]
    expected = (
        "fuzz",
        [
            {"role": "user", "content": "foo"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "thought"},
                    {
                        "type": "tool_use",
                        "name": "bar",
                        "id": "1",
                        "input": {"baz": "BUZZ"},  # tool_calls value preferred.
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": "blurb",
                        "tool_use_id": "1",
                        "is_error": False,
                    }
                ],
            },
        ],
    )
    actual = _format_messages(messages)
    assert expected == actual


def test__format_messages_with_cache_control() -> None:
    messages = [
        SystemMessage(
            [
                {"type": "text", "text": "foo", "cache_control": {"type": "ephemeral"}},
            ]
        ),
        HumanMessage(
            [
                {"type": "text", "text": "foo", "cache_control": {"type": "ephemeral"}},
                {
                    "type": "text",
                    "text": "foo",
                },
            ]
        ),
    ]
    expected_system = [
        {"type": "text", "text": "foo", "cache_control": {"type": "ephemeral"}}
    ]
    expected_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "foo", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "foo"},
            ],
        }
    ]
    actual_system, actual_messages = _format_messages(messages)
    assert expected_system == actual_system
    assert expected_messages == actual_messages


def test__format_messages_with_multiple_system() -> None:
    messages = [
        HumanMessage("baz"),
        SystemMessage("bar"),
        SystemMessage("baz"),
        SystemMessage(
            [
                {"type": "text", "text": "foo", "cache_control": {"type": "ephemeral"}},
            ]
        ),
    ]
    expected_system = [
        {"type": "text", "text": "bar"},
        {"type": "text", "text": "baz"},
        {"type": "text", "text": "foo", "cache_control": {"type": "ephemeral"}},
    ]
    expected_messages = [{"role": "user", "content": "baz"}]
    actual_system, actual_messages = _format_messages(messages)
    assert expected_system == actual_system
    assert expected_messages == actual_messages


def test_anthropic_api_key_is_secret_string() -> None:
    """Test that the API key is stored as a SecretStr."""
    chat_model = ChatAnthropic(  # type: ignore[call-arg, call-arg]
        model="claude-3-opus-20240229",
        anthropic_api_key="secret-api-key",
    )
    assert isinstance(chat_model.anthropic_api_key, SecretStr)


def test_anthropic_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test that the API key is masked when passed from an environment variable."""
    monkeypatch.setenv("ANTHROPIC_API_KEY ", "secret-api-key")
    chat_model = ChatAnthropic(  # type: ignore[call-arg]
        model="claude-3-opus-20240229",
    )
    print(chat_model.anthropic_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_anthropic_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test that the API key is masked when passed via the constructor."""
    chat_model = ChatAnthropic(  # type: ignore[call-arg, call-arg]
        model="claude-3-opus-20240229",
        anthropic_api_key="secret-api-key",
    )
    print(chat_model.anthropic_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_anthropic_uses_actual_secret_value_from_secretstr() -> None:
    """Test that the actual secret value is correctly retrieved."""
    chat_model = ChatAnthropic(  # type: ignore[call-arg, call-arg]
        model="claude-3-opus-20240229",
        anthropic_api_key="secret-api-key",
    )
    assert (
        cast(SecretStr, chat_model.anthropic_api_key).get_secret_value()
        == "secret-api-key"
    )


class GetWeather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


def test_anthropic_bind_tools_tool_choice() -> None:
    chat_model = ChatAnthropic(  # type: ignore[call-arg, call-arg]
        model="claude-3-opus-20240229",
        anthropic_api_key="secret-api-key",
    )
    chat_model_with_tools = chat_model.bind_tools(
        [GetWeather], tool_choice={"type": "tool", "name": "GetWeather"}
    )
    assert cast(RunnableBinding, chat_model_with_tools).kwargs["tool_choice"] == {
        "type": "tool",
        "name": "GetWeather",
    }
    chat_model_with_tools = chat_model.bind_tools(
        [GetWeather], tool_choice="GetWeather"
    )
    assert cast(RunnableBinding, chat_model_with_tools).kwargs["tool_choice"] == {
        "type": "tool",
        "name": "GetWeather",
    }
    chat_model_with_tools = chat_model.bind_tools([GetWeather], tool_choice="auto")
    assert cast(RunnableBinding, chat_model_with_tools).kwargs["tool_choice"] == {
        "type": "auto"
    }
    chat_model_with_tools = chat_model.bind_tools([GetWeather], tool_choice="any")
    assert cast(RunnableBinding, chat_model_with_tools).kwargs["tool_choice"] == {
        "type": "any"
    }
