"""Test chat model integration."""

from __future__ import annotations

import os
from typing import Any, Callable, Literal, Optional, cast
from unittest.mock import MagicMock, patch

import anthropic
import pytest
from anthropic.types import Message, TextBlock, Usage
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableBinding
from langchain_core.tools import BaseTool
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run
from pydantic import BaseModel, Field, SecretStr
from pytest import CaptureFixture, MonkeyPatch

from langchain_anthropic import ChatAnthropic
from langchain_anthropic.chat_models import (
    _create_usage_metadata,
    _format_image,
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
        assert cast("SecretStr", model.anthropic_api_key).get_secret_value() == "xyz"
        assert model.default_request_timeout == 2.0
        assert model.anthropic_api_url == "https://api.anthropic.com"


def test_anthropic_client_caching() -> None:
    """Test that the OpenAI client is cached."""
    llm1 = ChatAnthropic(model="claude-3-5-sonnet-latest")
    llm2 = ChatAnthropic(model="claude-3-5-sonnet-latest")
    assert llm1._client._client is llm2._client._client

    llm3 = ChatAnthropic(model="claude-3-5-sonnet-latest", base_url="foo")
    assert llm1._client._client is not llm3._client._client

    llm4 = ChatAnthropic(model="claude-3-5-sonnet-latest", timeout=None)
    assert llm1._client._client is llm4._client._client

    llm5 = ChatAnthropic(model="claude-3-5-sonnet-latest", timeout=3)
    assert llm1._client._client is not llm5._client._client


def test_anthropic_proxy_support() -> None:
    """Test that both sync and async clients support proxy configuration."""
    proxy_url = "http://proxy.example.com:8080"

    # Test sync client with proxy
    llm_sync = ChatAnthropic(
        model="claude-3-5-sonnet-latest", anthropic_proxy=proxy_url
    )
    sync_client = llm_sync._client
    assert sync_client is not None

    # Test async client with proxy - this should not raise TypeError
    async_client = llm_sync._async_client
    assert async_client is not None

    # Test that clients with different proxy settings are not cached together
    llm_no_proxy = ChatAnthropic(model="claude-3-5-sonnet-latest")
    llm_with_proxy = ChatAnthropic(
        model="claude-3-5-sonnet-latest", anthropic_proxy=proxy_url
    )

    # Different proxy settings should result in different cached clients
    assert llm_no_proxy._client._client is not llm_with_proxy._client._client


def test_anthropic_proxy_from_environment() -> None:
    """Test that proxy can be set from ANTHROPIC_PROXY environment variable."""
    proxy_url = "http://env-proxy.example.com:8080"

    # Test with environment variable set
    with patch.dict(os.environ, {"ANTHROPIC_PROXY": proxy_url}):
        llm = ChatAnthropic(model="claude-3-5-sonnet-latest")
        assert llm.anthropic_proxy == proxy_url

        # Should be able to create clients successfully
        sync_client = llm._client
        async_client = llm._async_client
        assert sync_client is not None
        assert async_client is not None

    # Test that explicit parameter overrides environment variable
    with patch.dict(os.environ, {"ANTHROPIC_PROXY": "http://env-proxy.com"}):
        explicit_proxy = "http://explicit-proxy.com"
        llm = ChatAnthropic(
            model="claude-3-5-sonnet-latest", anthropic_proxy=explicit_proxy
        )
        assert llm.anthropic_proxy == explicit_proxy


def test_set_default_max_tokens() -> None:
    """Test the set_default_max_tokens function."""
    # Test claude-opus-4 models
    llm = ChatAnthropic(model="claude-opus-4-20250514", anthropic_api_key="test")
    assert llm.max_tokens == 32000

    # Test claude-sonnet-4 models
    llm = ChatAnthropic(model="claude-sonnet-4-latest", anthropic_api_key="test")
    assert llm.max_tokens == 64000

    # Test claude-3-7-sonnet models
    llm = ChatAnthropic(model="claude-3-7-sonnet-latest", anthropic_api_key="test")
    assert llm.max_tokens == 64000

    # Test claude-3-5-sonnet models
    llm = ChatAnthropic(model="claude-3-5-sonnet-latest", anthropic_api_key="test")
    assert llm.max_tokens == 8192

    # Test claude-3-5-haiku models
    llm = ChatAnthropic(model="claude-3-5-haiku-latest", anthropic_api_key="test")
    assert llm.max_tokens == 8192

    # Test claude-3-haiku models (should default to 4096)
    llm = ChatAnthropic(model="claude-3-haiku-latest", anthropic_api_key="test")
    assert llm.max_tokens == 4096

    # Test that existing max_tokens values are preserved
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-latest", max_tokens=2048, anthropic_api_key="test"
    )
    assert llm.max_tokens == 2048

    # Test that explicitly set max_tokens values are preserved
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-latest", max_tokens=4096, anthropic_api_key="test"
    )
    assert llm.max_tokens == 4096


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
        response_metadata={"model_provider": "anthropic"},
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
        response_metadata={"model_provider": "anthropic"},
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
        model="claude-3-opus-20240229",
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
        cast("SecretStr", chat_model.anthropic_api_key).get_secret_value()
        == "secret-api-key"
    )


class GetWeather(BaseModel):
    """Get the current weather in a given location."""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


def test_anthropic_bind_tools_tool_choice() -> None:
    chat_model = ChatAnthropic(  # type: ignore[call-arg, call-arg]
        model="claude-3-opus-20240229",
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
    llm = ChatAnthropic(model="claude-3-5-haiku-latest")

    class SampleModel(BaseModel):
        sample_field: str

    _ = llm.with_structured_output(SampleModel.model_json_schema())


def test_get_num_tokens_from_messages_passes_kwargs() -> None:
    """Test that get_num_tokens_from_messages passes kwargs to the model."""
    llm = ChatAnthropic(model="claude-3-5-haiku-latest")

    with patch.object(anthropic, "Client") as _client:
        llm.get_num_tokens_from_messages([HumanMessage("foo")], foo="bar")

    assert _client.return_value.messages.count_tokens.call_args.kwargs["foo"] == "bar"

    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",
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
        input_tokens: Optional[int] = None
        output_tokens: Optional[int] = None
        cache_read_input_tokens: Optional[int] = None
        cache_creation_input_tokens: Optional[int] = None

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
        model="claude-sonnet-4-20250514",
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
    llm = ChatAnthropic(model="claude-3-5-haiku-latest")

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
        model="claude-sonnet-4-5-20250929",  # type: ignore[call-arg]
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
    llm = ChatAnthropic(model="claude-3-5-haiku-latest")

    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "anthropic",
        "ls_model_type": "chat",
        "ls_model_name": "claude-3-5-haiku-latest",
        "ls_max_tokens": 8192,
        "ls_temperature": None,
    }

    ls_params = llm._get_ls_params(model="claude-opus-4-1-20250805")
    assert ls_params.get("ls_model_name") == "claude-opus-4-1-20250805"


def test_streaming_cache_token_reporting() -> None:
    """Test that cache tokens are properly reported in streaming events."""
    from unittest.mock import MagicMock

    from anthropic.types import MessageDeltaUsage

    from langchain_anthropic.chat_models import _make_message_chunk_from_anthropic_event

    # Create a mock message_start event
    mock_message = MagicMock()
    mock_message.model = "claude-3-sonnet-20240229"
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
