"""Test chat model integration."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, Literal, cast
from unittest.mock import MagicMock, patch

import anthropic
import pytest
from anthropic.types import Message, TextBlock, Usage
from blockbuster import blockbuster_ctx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableBinding
from langchain_core.tools import BaseTool
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run
from pydantic import BaseModel, Field, SecretStr, ValidationError
from pytest import CaptureFixture, MonkeyPatch

from langchain_anthropic import ChatAnthropic
from langchain_anthropic.chat_models import (
    _create_usage_metadata,
    _format_image,
    _format_messages,
    _is_builtin_tool,
    _merge_messages,
    convert_to_anthropic_tool,
)

os.environ["ANTHROPIC_API_KEY"] = "foo"

MODEL_NAME = "claude-sonnet-4-5-20250929"


def test_initialization() -> None:
    """Test chat model initialization."""
    for model in [
        ChatAnthropic(model_name=MODEL_NAME, api_key="xyz", timeout=2),  # type: ignore[arg-type, call-arg]
        ChatAnthropic(  # type: ignore[call-arg, call-arg, call-arg]
            model=MODEL_NAME,
            anthropic_api_key="xyz",
            default_request_timeout=2,
            base_url="https://api.anthropic.com",
        ),
    ]:
        assert model.model == MODEL_NAME
        assert cast("SecretStr", model.anthropic_api_key).get_secret_value() == "xyz"
        assert model.default_request_timeout == 2.0
        assert model.anthropic_api_url == "https://api.anthropic.com"


@pytest.mark.parametrize("async_api", [True, False])
def test_streaming_attribute_should_stream(async_api: bool) -> None:  # noqa: FBT001
    llm = ChatAnthropic(model=MODEL_NAME, streaming=True)
    assert llm._should_stream(async_api=async_api)


def test_anthropic_client_caching() -> None:
    """Test that the OpenAI client is cached."""
    llm1 = ChatAnthropic(model=MODEL_NAME)
    llm2 = ChatAnthropic(model=MODEL_NAME)
    assert llm1._client._client is llm2._client._client

    llm3 = ChatAnthropic(model=MODEL_NAME, base_url="foo")
    assert llm1._client._client is not llm3._client._client

    llm4 = ChatAnthropic(model=MODEL_NAME, timeout=None)
    assert llm1._client._client is llm4._client._client

    llm5 = ChatAnthropic(model=MODEL_NAME, timeout=3)
    assert llm1._client._client is not llm5._client._client


def test_anthropic_proxy_support() -> None:
    """Test that both sync and async clients support proxy configuration."""
    proxy_url = "http://proxy.example.com:8080"

    # Test sync client with proxy
    llm_sync = ChatAnthropic(model=MODEL_NAME, anthropic_proxy=proxy_url)
    sync_client = llm_sync._client
    assert sync_client is not None

    # Test async client with proxy - this should not raise TypeError
    async_client = llm_sync._async_client
    assert async_client is not None

    # Test that clients with different proxy settings are not cached together
    llm_no_proxy = ChatAnthropic(model=MODEL_NAME)
    llm_with_proxy = ChatAnthropic(model=MODEL_NAME, anthropic_proxy=proxy_url)

    # Different proxy settings should result in different cached clients
    assert llm_no_proxy._client._client is not llm_with_proxy._client._client


def test_anthropic_proxy_from_environment() -> None:
    """Test that proxy can be set from ANTHROPIC_PROXY environment variable."""
    proxy_url = "http://env-proxy.example.com:8080"

    # Test with environment variable set
    with patch.dict(os.environ, {"ANTHROPIC_PROXY": proxy_url}):
        llm = ChatAnthropic(model=MODEL_NAME)
        assert llm.anthropic_proxy == proxy_url

        # Should be able to create clients successfully
        sync_client = llm._client
        async_client = llm._async_client
        assert sync_client is not None
        assert async_client is not None

    # Test that explicit parameter overrides environment variable
    with patch.dict(os.environ, {"ANTHROPIC_PROXY": "http://env-proxy.com"}):
        explicit_proxy = "http://explicit-proxy.com"
        llm = ChatAnthropic(model=MODEL_NAME, anthropic_proxy=explicit_proxy)
        assert llm.anthropic_proxy == explicit_proxy


def test_set_default_max_tokens() -> None:
    """Test the set_default_max_tokens function."""
    # Test claude-sonnet-4-5 models
    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", anthropic_api_key="test")
    assert llm.max_tokens == 64000

    # Test claude-opus-4 models
    llm = ChatAnthropic(model="claude-opus-4-20250514", anthropic_api_key="test")
    assert llm.max_tokens == 32000

    # Test claude-sonnet-4 models
    llm = ChatAnthropic(model="claude-sonnet-4-20250514", anthropic_api_key="test")
    assert llm.max_tokens == 64000

    # Test claude-3-7-sonnet models
    llm = ChatAnthropic(model="claude-3-7-sonnet-20250219", anthropic_api_key="test")
    assert llm.max_tokens == 64000

    # Test claude-3-5-haiku models
    llm = ChatAnthropic(model="claude-3-5-haiku-20241022", anthropic_api_key="test")
    assert llm.max_tokens == 8192

    # Test claude-3-haiku models (should default to 4096)
    llm = ChatAnthropic(model="claude-3-haiku-20240307", anthropic_api_key="test")
    assert llm.max_tokens == 4096

    # Test that existing max_tokens values are preserved
    llm = ChatAnthropic(model=MODEL_NAME, max_tokens=2048, anthropic_api_key="test")
    assert llm.max_tokens == 2048

    # Test that explicitly set max_tokens values are preserved
    llm = ChatAnthropic(model=MODEL_NAME, max_tokens=4096, anthropic_api_key="test")
    assert llm.max_tokens == 4096


@pytest.mark.requires("anthropic")
def test_anthropic_model_name_param() -> None:
    llm = ChatAnthropic(model_name=MODEL_NAME)  # type: ignore[call-arg, call-arg]
    assert llm.model == MODEL_NAME


@pytest.mark.requires("anthropic")
def test_anthropic_model_param() -> None:
    llm = ChatAnthropic(model=MODEL_NAME)  # type: ignore[call-arg]
    assert llm.model == MODEL_NAME


@pytest.mark.requires("anthropic")
def test_anthropic_model_kwargs() -> None:
    llm = ChatAnthropic(model_name=MODEL_NAME, model_kwargs={"foo": "bar"})  # type: ignore[call-arg, call-arg]
    assert llm.model_kwargs == {"foo": "bar"}


@pytest.mark.requires("anthropic")
def test_anthropic_fields_in_model_kwargs() -> None:
    """Test that for backwards compatibility fields can be passed in as model_kwargs."""
    llm = ChatAnthropic(model=MODEL_NAME, model_kwargs={"max_tokens_to_sample": 5})  # type: ignore[call-arg]
    assert llm.max_tokens == 5
    llm = ChatAnthropic(model=MODEL_NAME, model_kwargs={"max_tokens": 5})  # type: ignore[call-arg]
    assert llm.max_tokens == 5


@pytest.mark.requires("anthropic")
def test_anthropic_incorrect_field() -> None:
    with pytest.warns(match="not default parameter"):
        llm = ChatAnthropic(model=MODEL_NAME, foo="bar")  # type: ignore[call-arg, call-arg]
    assert llm.model_kwargs == {"foo": "bar"}


@pytest.mark.requires("anthropic")
def test_anthropic_initialization() -> None:
    """Test anthropic initialization."""
    # Verify that chat anthropic can be initialized using a secret key provided
    # as a parameter rather than an environment variable.
    ChatAnthropic(model=MODEL_NAME, anthropic_api_key="test")  # type: ignore[call-arg, call-arg]


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
        response_metadata={"model_provider": "anthropic"},
    )
    llm = ChatAnthropic(model=MODEL_NAME, anthropic_api_key="test")  # type: ignore[call-arg, call-arg]
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
        response_metadata={"model_provider": "anthropic"},
    )

    llm = ChatAnthropic(model=MODEL_NAME, anthropic_api_key="test")  # type: ignore[call-arg, call-arg]
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
            ],
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
            ],
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
            ],
        ),
    ]
    actual = _merge_messages(messages)
    assert expected == actual

    # Test tool message case
    messages = [
        ToolMessage("buz output", tool_call_id="1"),  # type: ignore[misc]
        ToolMessage(  # type: ignore[misc]
            content=[
                {"type": "tool_result", "content": "blah output", "tool_use_id": "2"},
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
            ],
        ),
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
            [{"type": "text", "text": "bar"}, {"type": "text", "text": "next thing"}],
        ),
    ]
    actual = _merge_messages(messages)
    assert expected == actual
    assert messages == original_messages


def test__format_image() -> None:
    url = "dummyimage.com/600x400/000/fff"
    with pytest.raises(ValueError):
        _format_image(url)


@pytest.fixture
def pydantic() -> type[BaseModel]:
    class dummy_function(BaseModel):  # noqa: N801
        """Dummy function."""

        arg1: int = Field(..., description="foo")
        arg2: Literal["bar", "baz"] = Field(..., description="one of 'bar', 'baz'")

    return dummy_function


@pytest.fixture
def function() -> Callable:
    def dummy_function(arg1: int, arg2: Literal["bar", "baz"]) -> None:
        """Dummy function.

        Args:
            arg1: foo
            arg2: one of 'bar', 'baz'

        """

    return dummy_function


@pytest.fixture
def dummy_tool() -> BaseTool:
    class Schema(BaseModel):
        arg1: int = Field(..., description="foo")
        arg2: Literal["bar", "baz"] = Field(..., description="one of 'bar', 'baz'")

    class DummyFunction(BaseTool):  # type: ignore[override]
        args_schema: type[BaseModel] = Schema
        name: str = "dummy_function"
        description: str = "Dummy function."

        def _run(self, *args: Any, **kwargs: Any) -> Any:
            pass

    return DummyFunction()


@pytest.fixture
def json_schema() -> dict:
    return {
        "title": "dummy_function",
        "description": "Dummy function.",
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


@pytest.fixture
def openai_function() -> dict:
    return {
        "name": "dummy_function",
        "description": "Dummy function.",
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
    pydantic: type[BaseModel],
    function: Callable,
    dummy_tool: BaseTool,
    json_schema: dict,
    openai_function: dict,
) -> None:
    expected = {
        "name": "dummy_function",
        "description": "Dummy function.",
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
        actual = convert_to_anthropic_tool(fn)
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
            },
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
                    },
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
                    },
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
                            },
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
                            },
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

    # Check handling of empty AIMessage
    empty_contents: list[str | list[str | dict]] = ["", []]
    for empty_content in empty_contents:
        ## Permit message in final position
        _, anthropic_messages = _format_messages([human, AIMessage(empty_content)])
        expected_messages = [
            {"role": "user", "content": "foo"},
            {"role": "assistant", "content": empty_content},
        ]
        assert expected_messages == anthropic_messages

        ## Remove message otherwise
        _, anthropic_messages = _format_messages(
            [human, AIMessage(empty_content), human]
        )
        expected_messages = [
            {"role": "user", "content": "foo"},
            {"role": "user", "content": "foo"},
        ]
        assert expected_messages == anthropic_messages

        actual = _format_messages(
            [system, human, ai, tool, AIMessage(empty_content), human]
        )
        assert actual[0] == "fuzz"
        assert [message["role"] for message in actual[1]] == [
            "user",
            "assistant",
            "user",
            "user",
        ]


def test__format_tool_use_block() -> None:
    # Test we correctly format tool_use blocks when there is no corresponding tool_call.
    message = AIMessage(
        [
            {
                "type": "tool_use",
                "name": "foo_1",
                "id": "1",
                "input": {"bar_1": "baz_1"},
            },
            {
                "type": "tool_use",
                "name": "foo_2",
                "id": "2",
                "input": {},
                "partial_json": '{"bar_2": "baz_2"}',
                "index": 1,
            },
        ]
    )
    result = _format_messages([message])
    expected = {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "name": "foo_1",
                "id": "1",
                "input": {"bar_1": "baz_1"},
            },
            {
                "type": "tool_use",
                "name": "foo_2",
                "id": "2",
                "input": {"bar_2": "baz_2"},
            },
        ],
    }
    assert result == (None, [expected])


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
                    },
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
                    },
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
                    },
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
            ],
        ),
        HumanMessage(
            [
                {"type": "text", "text": "foo", "cache_control": {"type": "ephemeral"}},
                {
                    "type": "text",
                    "text": "foo",
                },
            ],
        ),
    ]
    expected_system = [
        {"type": "text", "text": "foo", "cache_control": {"type": "ephemeral"}},
    ]
    expected_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "foo", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "foo"},
            ],
        },
    ]
    actual_system, actual_messages = _format_messages(messages)
    assert expected_system == actual_system
    assert expected_messages == actual_messages

    # Test standard multi-modal format (v0)
    messages = [
        HumanMessage(
            [
                {
                    "type": "text",
                    "text": "Summarize this document:",
                },
                {
                    "type": "file",
                    "source_type": "base64",
                    "mime_type": "application/pdf",
                    "data": "<base64 data>",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        ),
    ]
    actual_system, actual_messages = _format_messages(messages)
    assert actual_system is None
    expected_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Summarize this document:",
                },
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": "<base64 data>",
                    },
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        },
    ]
    assert actual_messages == expected_messages

    # Test standard multi-modal format (v1)
    messages = [
        HumanMessage(
            [
                {
                    "type": "text",
                    "text": "Summarize this document:",
                },
                {
                    "type": "file",
                    "mime_type": "application/pdf",
                    "base64": "<base64 data>",
                    "extras": {"cache_control": {"type": "ephemeral"}},
                },
            ],
        ),
    ]
    actual_system, actual_messages = _format_messages(messages)
    assert actual_system is None
    expected_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Summarize this document:",
                },
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": "<base64 data>",
                    },
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        },
    ]
    assert actual_messages == expected_messages

    # Test standard multi-modal format (v1, unpacked extras)
    messages = [
        HumanMessage(
            [
                {
                    "type": "text",
                    "text": "Summarize this document:",
                },
                {
                    "type": "file",
                    "mime_type": "application/pdf",
                    "base64": "<base64 data>",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        ),
    ]
    actual_system, actual_messages = _format_messages(messages)
    assert actual_system is None
    expected_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Summarize this document:",
                },
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": "<base64 data>",
                    },
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        },
    ]
    assert actual_messages == expected_messages

    # Also test file inputs
    ## Images
    for block in [
        # v1
        {
            "type": "image",
            "file_id": "abc123",
        },
        # v0
        {
            "type": "image",
            "source_type": "id",
            "id": "abc123",
        },
    ]:
        messages = [
            HumanMessage(
                [
                    {
                        "type": "text",
                        "text": "Summarize this image:",
                    },
                    block,
                ],
            ),
        ]
        actual_system, actual_messages = _format_messages(messages)
        assert actual_system is None
        expected_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Summarize this image:",
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "file",
                            "file_id": "abc123",
                        },
                    },
                ],
            },
        ]
        assert actual_messages == expected_messages

    ## Documents
    for block in [
        # v1
        {
            "type": "file",
            "file_id": "abc123",
        },
        # v0
        {
            "type": "file",
            "source_type": "id",
            "id": "abc123",
        },
    ]:
        messages = [
            HumanMessage(
                [
                    {
                        "type": "text",
                        "text": "Summarize this document:",
                    },
                    block,
                ],
            ),
        ]
        actual_system, actual_messages = _format_messages(messages)
        assert actual_system is None
        expected_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Summarize this document:",
                    },
                    {
                        "type": "document",
                        "source": {
                            "type": "file",
                            "file_id": "abc123",
                        },
                    },
                ],
            },
        ]
        assert actual_messages == expected_messages


def test__format_messages_with_citations() -> None:
    input_messages = [
        HumanMessage(
            content=[
                {
                    "type": "file",
                    "source_type": "text",
                    "text": "The grass is green. The sky is blue.",
                    "mime_type": "text/plain",
                    "citations": {"enabled": True},
                },
                {"type": "text", "text": "What color is the grass and sky?"},
            ],
        ),
    ]
    expected_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "text",
                        "media_type": "text/plain",
                        "data": "The grass is green. The sky is blue.",
                    },
                    "citations": {"enabled": True},
                },
                {"type": "text", "text": "What color is the grass and sky?"},
            ],
        },
    ]
    actual_system, actual_messages = _format_messages(input_messages)
    assert actual_system is None
    assert actual_messages == expected_messages


def test__format_messages_openai_image_format() -> None:
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Can you highlight the differences between these two images?",
            },
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,<base64 data>"},
            },
            {
                "type": "image_url",
                "image_url": {"url": "https://<image url>"},
            },
        ],
    )
    actual_system, actual_messages = _format_messages([message])
    assert actual_system is None
    expected_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Can you highlight the differences between these two images?"
                    ),
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "<base64 data>",
                    },
                },
                {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": "https://<image url>",
                    },
                },
            ],
        },
    ]
    assert actual_messages == expected_messages


def test__format_messages_with_multiple_system() -> None:
    messages = [
        HumanMessage("baz"),
        SystemMessage("bar"),
        SystemMessage("baz"),
        SystemMessage(
            [
                {"type": "text", "text": "foo", "cache_control": {"type": "ephemeral"}},
            ],
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
        model=MODEL_NAME,
        anthropic_api_key="secret-api-key",
    )
    assert isinstance(chat_model.anthropic_api_key, SecretStr)


def test_anthropic_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture,
) -> None:
    """Test that the API key is masked when passed from an environment variable."""
    monkeypatch.setenv("ANTHROPIC_API_KEY ", "secret-api-key")
    chat_model = ChatAnthropic(  # type: ignore[call-arg]
        model=MODEL_NAME,
    )
    print(chat_model.anthropic_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_anthropic_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test that the API key is masked when passed via the constructor."""
    chat_model = ChatAnthropic(  # type: ignore[call-arg, call-arg]
        model=MODEL_NAME,
        anthropic_api_key="secret-api-key",
    )
    print(chat_model.anthropic_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_anthropic_uses_actual_secret_value_from_secretstr() -> None:
    """Test that the actual secret value is correctly retrieved."""
    chat_model = ChatAnthropic(  # type: ignore[call-arg, call-arg]
        model=MODEL_NAME,
        anthropic_api_key="secret-api-key",
    )
    assert (
        cast("SecretStr", chat_model.anthropic_api_key).get_secret_value()
        == "secret-api-key"
    )


class GetWeather(BaseModel):
    """Get the current weather in a given location."""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


def test_anthropic_bind_tools_tool_choice() -> None:
    chat_model = ChatAnthropic(  # type: ignore[call-arg, call-arg]
        model=MODEL_NAME,
        anthropic_api_key="secret-api-key",
    )
    chat_model_with_tools = chat_model.bind_tools(
        [GetWeather],
        tool_choice={"type": "tool", "name": "GetWeather"},
    )
    assert cast("RunnableBinding", chat_model_with_tools).kwargs["tool_choice"] == {
        "type": "tool",
        "name": "GetWeather",
    }
    chat_model_with_tools = chat_model.bind_tools(
        [GetWeather],
        tool_choice="GetWeather",
    )
    assert cast("RunnableBinding", chat_model_with_tools).kwargs["tool_choice"] == {
        "type": "tool",
        "name": "GetWeather",
    }
    chat_model_with_tools = chat_model.bind_tools([GetWeather], tool_choice="auto")
    assert cast("RunnableBinding", chat_model_with_tools).kwargs["tool_choice"] == {
        "type": "auto",
    }
    chat_model_with_tools = chat_model.bind_tools([GetWeather], tool_choice="any")
    assert cast("RunnableBinding", chat_model_with_tools).kwargs["tool_choice"] == {
        "type": "any",
    }


def test_optional_description() -> None:
    llm = ChatAnthropic(model=MODEL_NAME)

    class SampleModel(BaseModel):
        sample_field: str

    _ = llm.with_structured_output(SampleModel.model_json_schema())


def test_get_num_tokens_from_messages_passes_kwargs() -> None:
    """Test that get_num_tokens_from_messages passes kwargs to the model."""
    llm = ChatAnthropic(model=MODEL_NAME)

    with patch.object(anthropic, "Client") as _client:
        llm.get_num_tokens_from_messages([HumanMessage("foo")], foo="bar")

    assert _client.return_value.messages.count_tokens.call_args.kwargs["foo"] == "bar"

    llm = ChatAnthropic(
        model=MODEL_NAME,
        betas=["context-management-2025-06-27"],
        context_management={"edits": [{"type": "clear_tool_uses_20250919"}]},
    )
    with patch.object(anthropic, "Client") as _client:
        llm.get_num_tokens_from_messages([HumanMessage("foo")])

    call_args = _client.return_value.beta.messages.count_tokens.call_args.kwargs
    assert call_args["betas"] == ["context-management-2025-06-27"]
    assert call_args["context_management"] == {
        "edits": [{"type": "clear_tool_uses_20250919"}]
    }


def test_usage_metadata_standardization() -> None:
    class UsageModel(BaseModel):
        input_tokens: int = 10
        output_tokens: int = 5
        cache_read_input_tokens: int = 3
        cache_creation_input_tokens: int = 2

    # Happy path
    usage = UsageModel()
    result = _create_usage_metadata(usage)
    assert result["input_tokens"] == 15  # 10 + 3 + 2
    assert result["output_tokens"] == 5
    assert result["total_tokens"] == 20
    assert result.get("input_token_details") == {"cache_read": 3, "cache_creation": 2}

    # Null input and output tokens
    class UsageModelNulls(BaseModel):
        input_tokens: int | None = None
        output_tokens: int | None = None
        cache_read_input_tokens: int | None = None
        cache_creation_input_tokens: int | None = None

    usage_nulls = UsageModelNulls()
    result = _create_usage_metadata(usage_nulls)
    assert result["input_tokens"] == 0
    assert result["output_tokens"] == 0
    assert result["total_tokens"] == 0

    # Test missing fields
    class UsageModelMissing(BaseModel):
        pass

    usage_missing = UsageModelMissing()
    result = _create_usage_metadata(usage_missing)
    assert result["input_tokens"] == 0
    assert result["output_tokens"] == 0
    assert result["total_tokens"] == 0


class FakeTracer(BaseTracer):
    """Fake tracer to capture inputs to `chat_model_start`."""

    def __init__(self) -> None:
        super().__init__()
        self.chat_model_start_inputs: list = []

    def _persist_run(self, run: Run) -> None:
        """Persist a run."""

    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> Run:
        self.chat_model_start_inputs.append({"args": args, "kwargs": kwargs})
        return super().on_chat_model_start(*args, **kwargs)


def test_mcp_tracing() -> None:
    # Test we exclude sensitive information from traces
    mcp_servers = [
        {
            "type": "url",
            "url": "https://mcp.deepwiki.com/mcp",
            "name": "deepwiki",
            "authorization_token": "PLACEHOLDER",
        },
    ]

    llm = ChatAnthropic(
        model=MODEL_NAME,
        betas=["mcp-client-2025-04-04"],
        mcp_servers=mcp_servers,
    )

    tracer = FakeTracer()
    mock_client = MagicMock()

    def mock_create(*args: Any, **kwargs: Any) -> Message:
        return Message(
            id="foo",
            content=[TextBlock(type="text", text="bar")],
            model="baz",
            role="assistant",
            stop_reason=None,
            stop_sequence=None,
            usage=Usage(input_tokens=2, output_tokens=1),
            type="message",
        )

    mock_client.messages.create = mock_create
    input_message = HumanMessage("Test query")
    with patch.object(llm, "_client", mock_client):
        _ = llm.invoke([input_message], config={"callbacks": [tracer]})

    # Test headers are not traced
    assert len(tracer.chat_model_start_inputs) == 1
    assert "PLACEHOLDER" not in str(tracer.chat_model_start_inputs)

    # Test headers are correctly propagated to request
    payload = llm._get_request_payload([input_message])
    assert payload["mcp_servers"][0]["authorization_token"] == "PLACEHOLDER"  # noqa: S105


def test_cache_control_kwarg() -> None:
    llm = ChatAnthropic(model=MODEL_NAME)

    messages = [HumanMessage("foo"), AIMessage("bar"), HumanMessage("baz")]
    payload = llm._get_request_payload(messages)
    assert payload["messages"] == [
        {"role": "user", "content": "foo"},
        {"role": "assistant", "content": "bar"},
        {"role": "user", "content": "baz"},
    ]

    payload = llm._get_request_payload(messages, cache_control={"type": "ephemeral"})
    assert payload["messages"] == [
        {"role": "user", "content": "foo"},
        {"role": "assistant", "content": "bar"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "baz", "cache_control": {"type": "ephemeral"}}
            ],
        },
    ]

    messages = [
        HumanMessage("foo"),
        AIMessage("bar"),
        HumanMessage(
            content=[
                {"type": "text", "text": "baz"},
                {"type": "text", "text": "qux"},
            ]
        ),
    ]
    payload = llm._get_request_payload(messages, cache_control={"type": "ephemeral"})
    assert payload["messages"] == [
        {"role": "user", "content": "foo"},
        {"role": "assistant", "content": "bar"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "baz"},
                {"type": "text", "text": "qux", "cache_control": {"type": "ephemeral"}},
            ],
        },
    ]


def test_context_management_in_payload() -> None:
    llm = ChatAnthropic(
        model=MODEL_NAME,  # type: ignore[call-arg]
        betas=["context-management-2025-06-27"],
        context_management={"edits": [{"type": "clear_tool_uses_20250919"}]},
    )
    llm_with_tools = llm.bind_tools(
        [{"type": "web_search_20250305", "name": "web_search"}]
    )
    input_message = HumanMessage("Search for recent developments in AI")
    payload = llm_with_tools._get_request_payload([input_message])  # type: ignore[attr-defined]
    assert payload["context_management"] == {
        "edits": [{"type": "clear_tool_uses_20250919"}]
    }


def test_anthropic_model_params() -> None:
    llm = ChatAnthropic(model=MODEL_NAME)

    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "anthropic",
        "ls_model_type": "chat",
        "ls_model_name": MODEL_NAME,
        "ls_max_tokens": 64000,
        "ls_temperature": None,
    }

    ls_params = llm._get_ls_params(model=MODEL_NAME)
    assert ls_params.get("ls_model_name") == MODEL_NAME


def test_streaming_cache_token_reporting() -> None:
    """Test that cache tokens are properly reported in streaming events."""
    from unittest.mock import MagicMock

    from anthropic.types import MessageDeltaUsage

    from langchain_anthropic.chat_models import _make_message_chunk_from_anthropic_event

    # Create a mock message_start event
    mock_message = MagicMock()
    mock_message.model = MODEL_NAME
    mock_message.usage.input_tokens = 100
    mock_message.usage.output_tokens = 0
    mock_message.usage.cache_read_input_tokens = 25
    mock_message.usage.cache_creation_input_tokens = 10

    message_start_event = MagicMock()
    message_start_event.type = "message_start"
    message_start_event.message = mock_message

    # Create a mock message_delta event with complete usage info
    mock_delta_usage = MessageDeltaUsage(
        output_tokens=50,
        input_tokens=100,
        cache_read_input_tokens=25,
        cache_creation_input_tokens=10,
    )

    mock_delta = MagicMock()
    mock_delta.stop_reason = "end_turn"
    mock_delta.stop_sequence = None

    message_delta_event = MagicMock()
    message_delta_event.type = "message_delta"
    message_delta_event.usage = mock_delta_usage
    message_delta_event.delta = mock_delta

    # Test message_start event
    start_chunk, _ = _make_message_chunk_from_anthropic_event(
        message_start_event,
        stream_usage=True,
        coerce_content_to_string=True,
        block_start_event=None,
    )

    # Test message_delta event - should contain complete usage metadata (w/ cache)
    delta_chunk, _ = _make_message_chunk_from_anthropic_event(
        message_delta_event,
        stream_usage=True,
        coerce_content_to_string=True,
        block_start_event=None,
    )

    # Verify message_delta has complete usage_metadata including cache tokens
    assert start_chunk is not None, "message_start should produce a chunk"
    assert getattr(start_chunk, "usage_metadata", None) is None, (
        "message_start should not have usage_metadata"
    )
    assert delta_chunk is not None, "message_delta should produce a chunk"
    assert delta_chunk.usage_metadata is not None, (
        "message_delta should have usage_metadata"
    )
    assert "input_token_details" in delta_chunk.usage_metadata
    input_details = delta_chunk.usage_metadata["input_token_details"]
    assert input_details.get("cache_read") == 25
    assert input_details.get("cache_creation") == 10

    # Verify totals are correct: 100 base + 25 cache_read + 10 cache_creation = 135
    assert delta_chunk.usage_metadata["input_tokens"] == 135
    assert delta_chunk.usage_metadata["output_tokens"] == 50
    assert delta_chunk.usage_metadata["total_tokens"] == 185


def test_strict_tool_use() -> None:
    model = ChatAnthropic(
        model=MODEL_NAME,  # type: ignore[call-arg]
        betas=["structured-outputs-2025-11-13"],
    )

    def get_weather(location: str, unit: Literal["C", "F"]) -> str:
        """Get the weather at a location."""
        return "75 degrees Fahrenheit."

    model_with_tools = model.bind_tools([get_weather], strict=True)

    tool_definition = model_with_tools.kwargs["tools"][0]  # type: ignore[attr-defined]
    assert tool_definition["strict"] is True


def test_beta_merging_with_response_format() -> None:
    """Test that structured-outputs beta is merged with existing betas."""

    class Person(BaseModel):
        """Person data."""

        name: str
        age: int

    # Auto-inject structured-outputs beta with no others specified
    model = ChatAnthropic(model=MODEL_NAME)
    payload = model._get_request_payload(
        "Test query",
        response_format=Person.model_json_schema(),
    )
    assert payload["betas"] == ["structured-outputs-2025-11-13"]

    # Merge structured-outputs beta if other betas are present
    model = ChatAnthropic(
        model=MODEL_NAME,
        betas=["mcp-client-2025-04-04"],
    )
    payload = model._get_request_payload(
        "Test query",
        response_format=Person.model_json_schema(),
    )
    assert payload["betas"] == [
        "mcp-client-2025-04-04",
        "structured-outputs-2025-11-13",
    ]

    # Structured-outputs beta already present - don't duplicate
    model = ChatAnthropic(
        model=MODEL_NAME,
        betas=[
            "mcp-client-2025-04-04",
            "structured-outputs-2025-11-13",
        ],
    )
    payload = model._get_request_payload(
        "Test query",
        response_format=Person.model_json_schema(),
    )
    assert payload["betas"] == [
        "mcp-client-2025-04-04",
        "structured-outputs-2025-11-13",
    ]

    # No response_format - betas should not be modified
    model = ChatAnthropic(
        model=MODEL_NAME,
        betas=["mcp-client-2025-04-04"],
    )
    payload = model._get_request_payload("Test query")
    assert payload["betas"] == ["mcp-client-2025-04-04"]


def test_beta_merging_with_strict_tool_use() -> None:
    """Test beta merging for strict tools."""

    def get_weather(location: str) -> str:
        """Get the weather at a location."""
        return "Sunny"

    # Auto-inject structured-outputs beta with no others specified
    model = ChatAnthropic(model=MODEL_NAME)  # type: ignore[call-arg]
    model_with_tools = model.bind_tools([get_weather], strict=True)
    payload = model_with_tools._get_request_payload(  # type: ignore[attr-defined]
        "What's the weather?",
        **model_with_tools.kwargs,  # type: ignore[attr-defined]
    )
    assert payload["betas"] == ["structured-outputs-2025-11-13"]

    # Merge structured-outputs beta if other betas are present
    model = ChatAnthropic(
        model=MODEL_NAME,  # type: ignore[call-arg]
        betas=["mcp-client-2025-04-04"],
    )
    model_with_tools = model.bind_tools([get_weather], strict=True)
    payload = model_with_tools._get_request_payload(  # type: ignore[attr-defined]
        "What's the weather?",
        **model_with_tools.kwargs,  # type: ignore[attr-defined]
    )
    assert payload["betas"] == [
        "mcp-client-2025-04-04",
        "structured-outputs-2025-11-13",
    ]

    # Structured-outputs beta already present - don't duplicate
    model = ChatAnthropic(
        model=MODEL_NAME,  # type: ignore[call-arg]
        betas=[
            "mcp-client-2025-04-04",
            "structured-outputs-2025-11-13",
        ],
    )
    model_with_tools = model.bind_tools([get_weather], strict=True)
    payload = model_with_tools._get_request_payload(  # type: ignore[attr-defined]
        "What's the weather?",
        **model_with_tools.kwargs,  # type: ignore[attr-defined]
    )
    assert payload["betas"] == [
        "mcp-client-2025-04-04",
        "structured-outputs-2025-11-13",
    ]

    # No strict tools - betas should not be modified
    model = ChatAnthropic(
        model=MODEL_NAME,  # type: ignore[call-arg]
        betas=["mcp-client-2025-04-04"],
    )
    model_with_tools = model.bind_tools([get_weather], strict=False)
    payload = model_with_tools._get_request_payload(  # type: ignore[attr-defined]
        "What's the weather?",
        **model_with_tools.kwargs,  # type: ignore[attr-defined]
    )
    assert payload["betas"] == ["mcp-client-2025-04-04"]


def test_auto_append_betas_for_tool_types() -> None:
    """Test that betas are automatically appended based on tool types."""
    # Test web_fetch_20250910 auto-appends web-fetch-2025-09-10
    model = ChatAnthropic(model=MODEL_NAME)  # type: ignore[call-arg]
    tool = {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 3}
    model_with_tools = model.bind_tools([tool])
    payload = model_with_tools._get_request_payload(  # type: ignore[attr-defined]
        "test",
        **model_with_tools.kwargs,  # type: ignore[attr-defined]
    )
    assert payload["betas"] == ["web-fetch-2025-09-10"]

    # Test code_execution_20250522 auto-appends code-execution-2025-05-22
    model = ChatAnthropic(model=MODEL_NAME)  # type: ignore[call-arg]
    tool = {"type": "code_execution_20250522", "name": "code_execution"}
    model_with_tools = model.bind_tools([tool])
    payload = model_with_tools._get_request_payload(  # type: ignore[attr-defined]
        "test",
        **model_with_tools.kwargs,  # type: ignore[attr-defined]
    )
    assert payload["betas"] == ["code-execution-2025-05-22"]

    # Test memory_20250818 auto-appends context-management-2025-06-27
    model = ChatAnthropic(model=MODEL_NAME)  # type: ignore[call-arg]
    tool = {"type": "memory_20250818", "name": "memory"}
    model_with_tools = model.bind_tools([tool])
    payload = model_with_tools._get_request_payload(  # type: ignore[attr-defined]
        "test",
        **model_with_tools.kwargs,  # type: ignore[attr-defined]
    )
    assert payload["betas"] == ["context-management-2025-06-27"]

    # Test merging with existing betas
    model = ChatAnthropic(
        model=MODEL_NAME,
        betas=["mcp-client-2025-04-04"],  # type: ignore[call-arg]
    )
    tool = {"type": "web_fetch_20250910", "name": "web_fetch"}
    model_with_tools = model.bind_tools([tool])
    payload = model_with_tools._get_request_payload(  # type: ignore[attr-defined]
        "test",
        **model_with_tools.kwargs,  # type: ignore[attr-defined]
    )
    assert payload["betas"] == ["mcp-client-2025-04-04", "web-fetch-2025-09-10"]

    # Test that it doesn't duplicate existing betas
    model = ChatAnthropic(
        model=MODEL_NAME,
        betas=["web-fetch-2025-09-10"],  # type: ignore[call-arg]
    )
    tool = {"type": "web_fetch_20250910", "name": "web_fetch"}
    model_with_tools = model.bind_tools([tool])
    payload = model_with_tools._get_request_payload(  # type: ignore[attr-defined]
        "test",
        **model_with_tools.kwargs,  # type: ignore[attr-defined]
    )
    assert payload["betas"] == ["web-fetch-2025-09-10"]

    # Test multiple tools with different beta requirements
    model = ChatAnthropic(model=MODEL_NAME)  # type: ignore[call-arg]
    tools = [
        {"type": "web_fetch_20250910", "name": "web_fetch"},
        {"type": "code_execution_20250522", "name": "code_execution"},
    ]
    model_with_tools = model.bind_tools(tools)
    payload = model_with_tools._get_request_payload(  # type: ignore[attr-defined]
        "test",
        **model_with_tools.kwargs,  # type: ignore[attr-defined]
    )
    assert set(payload["betas"]) == {
        "web-fetch-2025-09-10",
        "code-execution-2025-05-22",
    }


def test_tool_search_is_builtin_tool() -> None:
    """Test that tool search tools are recognized as built-in tools."""
    # Test regex variant
    regex_tool = {
        "type": "tool_search_tool_regex_20251119",
        "name": "tool_search_tool_regex",
    }
    assert _is_builtin_tool(regex_tool)

    # Test BM25 variant
    bm25_tool = {
        "type": "tool_search_tool_bm25_20251119",
        "name": "tool_search_tool_bm25",
    }
    assert _is_builtin_tool(bm25_tool)

    # Test non-builtin tool
    regular_tool = {
        "name": "get_weather",
        "description": "Get weather",
        "input_schema": {"type": "object", "properties": {}},
    }
    assert not _is_builtin_tool(regular_tool)


def test_tool_search_beta_headers() -> None:
    """Test that tool search tools auto-append the correct beta headers."""
    # Test regex variant
    model = ChatAnthropic(model=MODEL_NAME)  # type: ignore[call-arg]
    regex_tool = {
        "type": "tool_search_tool_regex_20251119",
        "name": "tool_search_tool_regex",
    }
    model_with_tools = model.bind_tools([regex_tool])
    payload = model_with_tools._get_request_payload(  # type: ignore[attr-defined]
        "test",
        **model_with_tools.kwargs,  # type: ignore[attr-defined]
    )
    assert payload["betas"] == ["advanced-tool-use-2025-11-20"]

    # Test BM25 variant
    model = ChatAnthropic(model=MODEL_NAME)  # type: ignore[call-arg]
    bm25_tool = {
        "type": "tool_search_tool_bm25_20251119",
        "name": "tool_search_tool_bm25",
    }
    model_with_tools = model.bind_tools([bm25_tool])
    payload = model_with_tools._get_request_payload(  # type: ignore[attr-defined]
        "test",
        **model_with_tools.kwargs,  # type: ignore[attr-defined]
    )
    assert payload["betas"] == ["advanced-tool-use-2025-11-20"]

    # Test merging with existing betas
    model = ChatAnthropic(
        model=MODEL_NAME,
        betas=["mcp-client-2025-04-04"],  # type: ignore[call-arg]
    )
    model_with_tools = model.bind_tools([regex_tool])
    payload = model_with_tools._get_request_payload(  # type: ignore[attr-defined]
        "test",
        **model_with_tools.kwargs,  # type: ignore[attr-defined]
    )
    assert payload["betas"] == [
        "mcp-client-2025-04-04",
        "advanced-tool-use-2025-11-20",
    ]


def test_tool_search_with_deferred_tools() -> None:
    """Test that `defer_loading` works correctly with tool search."""
    llm = ChatAnthropic(
        model="claude-opus-4-5-20251101",  # type: ignore[call-arg]
    )

    # Create tools with defer_loading
    tools = [
        {
            "type": "tool_search_tool_bm25_20251119",
            "name": "tool_search_tool_bm25",
        },
        {
            "name": "calculator",
            "description": "Perform mathematical calculations",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression",
                    },
                },
                "required": ["expression"],
            },
            "defer_loading": True,
        },
    ]

    llm_with_tools = llm.bind_tools(tools)  # type: ignore[arg-type]

    # Verify the payload includes tools with defer_loading
    payload = llm_with_tools._get_request_payload(  # type: ignore[attr-defined]
        "test",
        **llm_with_tools.kwargs,  # type: ignore[attr-defined]
    )

    # Find the calculator tool in the payload
    calculator_tool = None
    for tool in payload["tools"]:
        if isinstance(tool, dict) and tool.get("name") == "calculator":
            calculator_tool = tool
            break

    assert calculator_tool is not None
    assert calculator_tool.get("defer_loading") is True


def test_tool_search_result_formatting() -> None:
    """Test that `tool_result` blocks with `tool_reference` are handled correctly."""
    # Tool search result with tool_reference blocks
    messages = [
        HumanMessage("What tools can help with weather?"),  # type: ignore[misc]
        AIMessage(  # type: ignore[misc]
            [
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_123",
                    "name": "tool_search_tool_regex",
                    "input": {"query": "weather"},
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "srvtoolu_123",
                    "content": [
                        {"type": "tool_reference", "tool_name": "get_weather"},
                        {"type": "tool_reference", "tool_name": "weather_forecast"},
                    ],
                },
            ],
        ),
    ]

    _, formatted = _format_messages(messages)

    # Verify the tool_result block is preserved correctly
    assistant_msg = formatted[1]
    assert assistant_msg["role"] == "assistant"

    # Find the tool_result block
    tool_result_block = None
    for block in assistant_msg["content"]:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            tool_result_block = block
            break

    assert tool_result_block is not None
    assert tool_result_block["tool_use_id"] == "srvtoolu_123"
    assert isinstance(tool_result_block["content"], list)
    assert len(tool_result_block["content"]) == 2
    assert tool_result_block["content"][0]["type"] == "tool_reference"
    assert tool_result_block["content"][0]["tool_name"] == "get_weather"
    assert tool_result_block["content"][1]["type"] == "tool_reference"
    assert tool_result_block["content"][1]["tool_name"] == "weather_forecast"


def test_auto_append_betas_for_mcp_servers() -> None:
    """Test that `mcp-client-2025-11-20` beta is automatically appended
    for `mcp_servers`."""
    model = ChatAnthropic(model=MODEL_NAME)  # type: ignore[call-arg]
    mcp_servers = [
        {
            "type": "url",
            "url": "https://mcp.example.com/mcp",
            "name": "example",
        }
    ]
    payload = model._get_request_payload(
        "Test query",
        mcp_servers=mcp_servers,  # type: ignore[arg-type]
    )
    assert payload["betas"] == ["mcp-client-2025-11-20"]
    assert payload["mcp_servers"] == mcp_servers

    # Test merging with existing betas
    model = ChatAnthropic(
        model=MODEL_NAME,
        betas=["context-management-2025-06-27"],
    )
    payload = model._get_request_payload(
        "Test query",
        mcp_servers=mcp_servers,  # type: ignore[arg-type]
    )
    assert payload["betas"] == [
        "context-management-2025-06-27",
        "mcp-client-2025-11-20",
    ]

    # Test that it doesn't duplicate if beta already present
    model = ChatAnthropic(
        model=MODEL_NAME,
        betas=["mcp-client-2025-11-20"],
    )
    payload = model._get_request_payload(
        "Test query",
        mcp_servers=mcp_servers,  # type: ignore[arg-type]
    )
    assert payload["betas"] == ["mcp-client-2025-11-20"]

    # Test with mcp_servers set on model initialization
    model = ChatAnthropic(
        model=MODEL_NAME,
        mcp_servers=mcp_servers,  # type: ignore[arg-type]
    )
    payload = model._get_request_payload("Test query")
    assert payload["betas"] == ["mcp-client-2025-11-20"]
    assert payload["mcp_servers"] == mcp_servers

    # Test with existing betas and mcp_servers on model initialization
    model = ChatAnthropic(
        model=MODEL_NAME,
        betas=["context-management-2025-06-27"],
        mcp_servers=mcp_servers,  # type: ignore[arg-type]
    )
    payload = model._get_request_payload("Test query")
    assert payload["betas"] == [
        "context-management-2025-06-27",
        "mcp-client-2025-11-20",
    ]

    # Test that beta is not appended when mcp_servers is None
    model = ChatAnthropic(model=MODEL_NAME)
    payload = model._get_request_payload("Test query")
    assert "betas" not in payload or payload["betas"] is None

    # Test combining mcp_servers with tool types that require betas
    model = ChatAnthropic(model=MODEL_NAME)
    tool = {"type": "web_fetch_20250910", "name": "web_fetch"}
    model_with_tools = model.bind_tools([tool])
    payload = model_with_tools._get_request_payload(  # type: ignore[attr-defined]
        "Test query",
        mcp_servers=mcp_servers,
        **model_with_tools.kwargs,  # type: ignore[attr-defined]
    )
    assert set(payload["betas"]) == {
        "web-fetch-2025-09-10",
        "mcp-client-2025-11-20",
    }


def test_profile() -> None:
    model = ChatAnthropic(model="claude-sonnet-4-20250514")
    assert model.profile
    assert not model.profile["structured_output"]

    model = ChatAnthropic(model="claude-sonnet-4-5")
    assert model.profile
    assert model.profile["structured_output"]
    assert model.profile["tool_calling"]

    # Test overwriting a field
    model.profile["tool_calling"] = False
    assert not model.profile["tool_calling"]

    # Test we didn't mutate
    model = ChatAnthropic(model="claude-sonnet-4-5")
    assert model.profile
    assert model.profile["tool_calling"]

    # Test passing in profile
    model = ChatAnthropic(model="claude-sonnet-4-5", profile={"tool_calling": False})
    assert model.profile == {"tool_calling": False}


async def test_model_profile_not_blocking() -> None:
    with blockbuster_ctx():
        model = ChatAnthropic(model="claude-sonnet-4-5")
        _ = model.profile


def test_effort_parameter_validation() -> None:
    """Test that effort parameter is validated correctly.

    The effort parameter is currently in beta and only supported by Claude Opus 4.5.
    """
    # Valid effort values should work
    model = ChatAnthropic(model="claude-opus-4-5-20251101", effort="high")
    assert model.effort == "high"

    model = ChatAnthropic(model="claude-opus-4-5-20251101", effort="medium")
    assert model.effort == "medium"

    model = ChatAnthropic(model="claude-opus-4-5-20251101", effort="low")
    assert model.effort == "low"

    # Invalid effort values should raise ValidationError
    with pytest.raises(ValidationError, match="Input should be"):
        ChatAnthropic(model="claude-opus-4-5-20251101", effort="invalid")  # type: ignore[arg-type]


def test_effort_populates_betas() -> None:
    """Test that effort parameter auto-populates required betas."""
    model = ChatAnthropic(model="claude-opus-4-5-20251101", effort="medium")
    assert model.effort == "medium"

    # Test that effort works with dated API ID
    payload = model._get_request_payload("Test query")
    assert payload["output_config"]["effort"] == "medium"
    assert "effort-2025-11-24" in payload["betas"]


def test_effort_in_output_config() -> None:
    """Test that effort can be specified in `output_config`."""
    # Test valid effort in output_config
    model = ChatAnthropic(
        model="claude-opus-4-5-20251101",
        output_config={"effort": "low"},
    )
    assert model.model_kwargs["output_config"] == {"effort": "low"}


def test_effort_priority() -> None:
    """Test that top-level effort takes precedence over `output_config`."""
    model = ChatAnthropic(
        model="claude-opus-4-5-20251101",
        effort="high",
        output_config={"effort": "low"},
    )

    # Top-level effort should take precedence in the payload
    payload = model._get_request_payload("Test query")
    assert payload["output_config"]["effort"] == "high"


def test_effort_beta_header_auto_append() -> None:
    """Test that effort beta header is automatically appended."""
    # Test with top-level effort parameter
    model = ChatAnthropic(model="claude-opus-4-5-20251101", effort="medium")
    payload = model._get_request_payload("Test query")
    assert "effort-2025-11-24" in payload["betas"]

    # Test with output_config
    model = ChatAnthropic(
        model="claude-opus-4-5-20251101",
        output_config={"effort": "low"},
    )
    payload = model._get_request_payload("Test query")
    assert "effort-2025-11-24" in payload["betas"]

    # Test that beta is not duplicated if already present
    model = ChatAnthropic(
        model="claude-opus-4-5-20251101",
        effort="high",
        betas=["effort-2025-11-24"],
    )
    payload = model._get_request_payload("Test query")
    assert payload["betas"].count("effort-2025-11-24") == 1

    # Test combining effort with other betas
    model = ChatAnthropic(
        model="claude-opus-4-5-20251101",
        effort="medium",
        betas=["context-1m-2025-08-07"],
    )
    payload = model._get_request_payload("Test query")
    assert set(payload["betas"]) == {
        "context-1m-2025-08-07",
        "effort-2025-11-24",
    }


def test_output_config_without_effort() -> None:
    """Test that output_config can be used without effort."""
    # output_config might have other fields in the future
    model = ChatAnthropic(
        model=MODEL_NAME,
        output_config={"some_future_param": "value"},
    )
    payload = model._get_request_payload("Test query")
    assert payload["output_config"] == {"some_future_param": "value"}
    # No effort beta should be added
    assert payload.get("betas") is None or "effort-2025-11-24" not in payload.get(
        "betas", []
    )


def test_extras_with_defer_loading() -> None:
    """Test that extras with `defer_loading` are merged into tool definitions."""
    from langchain_core.tools import tool

    @tool(extras={"defer_loading": True})
    def get_weather(location: str) -> str:
        """Get weather for a location."""
        return f"Weather in {location}"

    model = ChatAnthropic(model=MODEL_NAME)  # type: ignore[call-arg]
    model_with_tools = model.bind_tools([get_weather])

    # Get the payload to check if defer_loading was merged
    payload = model_with_tools._get_request_payload(  # type: ignore[attr-defined]
        "test",
        **model_with_tools.kwargs,  # type: ignore[attr-defined]
    )

    # Find the get_weather tool in the payload
    weather_tool = None
    for tool_def in payload["tools"]:
        if isinstance(tool_def, dict) and tool_def.get("name") == "get_weather":
            weather_tool = tool_def
            break

    assert weather_tool is not None
    assert weather_tool.get("defer_loading") is True


def test_extras_with_cache_control() -> None:
    """Test that extras with `cache_control` are merged into tool definitions."""
    from langchain_core.tools import tool

    @tool(extras={"cache_control": {"type": "ephemeral"}})
    def search_files(query: str) -> str:
        """Search files."""
        return f"Results for {query}"

    model = ChatAnthropic(model=MODEL_NAME)  # type: ignore[call-arg]
    model_with_tools = model.bind_tools([search_files])

    payload = model_with_tools._get_request_payload(  # type: ignore[attr-defined]
        "test",
        **model_with_tools.kwargs,  # type: ignore[attr-defined]
    )

    search_tool = None
    for tool_def in payload["tools"]:
        if isinstance(tool_def, dict) and tool_def.get("name") == "search_files":
            search_tool = tool_def
            break

    assert search_tool is not None
    assert search_tool.get("cache_control") == {"type": "ephemeral"}


def test_extras_with_input_examples() -> None:
    """Test that extras with `input_examples` are merged into tool definitions."""
    from langchain_core.tools import tool

    @tool(
        extras={
            "input_examples": [
                {"location": "San Francisco, CA", "unit": "fahrenheit"},
                {"location": "Tokyo, Japan", "unit": "celsius"},
            ]
        }
    )
    def get_weather(location: str, unit: str = "fahrenheit") -> str:
        """Get weather for a location."""
        return f"Weather in {location}"

    model = ChatAnthropic(model=MODEL_NAME)  # type: ignore[call-arg]
    model_with_tools = model.bind_tools([get_weather])

    payload = model_with_tools._get_request_payload(  # type: ignore[attr-defined]
        "test",
        **model_with_tools.kwargs,  # type: ignore[attr-defined]
    )

    weather_tool = None
    for tool_def in payload["tools"]:
        if isinstance(tool_def, dict) and tool_def.get("name") == "get_weather":
            weather_tool = tool_def
            break

    assert weather_tool is not None
    assert "input_examples" in weather_tool
    assert len(weather_tool["input_examples"]) == 2
    assert weather_tool["input_examples"][0] == {
        "location": "San Francisco, CA",
        "unit": "fahrenheit",
    }


def test_extras_with_multiple_fields() -> None:
    """Test that multiple extra fields can be specified together."""
    from langchain_core.tools import tool

    @tool(
        extras={
            "defer_loading": True,
            "cache_control": {"type": "ephemeral"},
            "input_examples": [{"query": "python files"}],
        }
    )
    def search_code(query: str) -> str:
        """Search code."""
        return f"Code for {query}"

    model = ChatAnthropic(model=MODEL_NAME)  # type: ignore[call-arg]
    model_with_tools = model.bind_tools([search_code])

    payload = model_with_tools._get_request_payload(  # type: ignore[attr-defined]
        "test",
        **model_with_tools.kwargs,  # type: ignore[attr-defined]
    )

    tool_def = None
    for t in payload["tools"]:
        if isinstance(t, dict) and t.get("name") == "search_code":
            tool_def = t
            break

    assert tool_def is not None
    assert tool_def.get("defer_loading") is True
    assert tool_def.get("cache_control") == {"type": "ephemeral"}
    assert "input_examples" in tool_def
