"""Module that contains tests for runnable.astream_events API."""

import asyncio
import sys
from collections.abc import AsyncIterator, Sequence
from itertools import cycle
from typing import Any, cast

import pytest
from pydantic import BaseModel
from typing_extensions import override

from langchain_core.callbacks import CallbackManagerForRetrieverRun, Callbacks
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.language_models import FakeStreamingListLLM, GenericFakeChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    ConfigurableField,
    RunnableConfig,
    RunnableLambda,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.schema import StreamEvent
from langchain_core.tools import tool
from tests.unit_tests.stubs import _any_id_ai_message, _any_id_ai_message_chunk


def _with_nulled_run_id(events: Sequence[StreamEvent]) -> list[StreamEvent]:
    """Removes the run IDs from events."""
    for event in events:
        assert "parent_ids" in event, "Parent IDs should be present in the event."
        assert event["parent_ids"] == [], "Parent IDs should be empty."

    return cast("list[StreamEvent]", [{**event, "run_id": ""} for event in events])


async def _as_async_iterator(iterable: list) -> AsyncIterator:
    """Converts an iterable into an async iterator."""
    for item in iterable:
        yield item


async def _collect_events(events: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    """Collect the events and remove the run ids."""
    materialized_events = [event async for event in events]
    events_ = _with_nulled_run_id(materialized_events)
    for event in events_:
        event["tags"] = sorted(event["tags"])
    return events_


def _assert_events_equal_allow_superset_metadata(events: list, expected: list) -> None:
    """Assert that the events are equal."""
    assert len(events) == len(expected)
    for i, (event, expected_event) in enumerate(zip(events, expected, strict=False)):
        # we want to allow a superset of metadata on each
        event_with_edited_metadata = {
            k: (
                v
                if k != "metadata"
                else {
                    metadata_k: metadata_v
                    for metadata_k, metadata_v in v.items()
                    if metadata_k in expected_event["metadata"]
                }
            )
            for k, v in event.items()
        }
        assert event_with_edited_metadata == expected_event, f"Event {i} did not match."


async def test_event_stream_with_simple_function_tool() -> None:
    """Test the event stream with a function and tool."""

    def foo(_: int) -> dict:
        """Foo."""
        return {"x": 5}

    @tool
    def get_docs(x: int) -> list[Document]:  # noqa: ARG001
        """Hello Doc."""
        return [Document(page_content="hello")]

    chain = RunnableLambda(foo) | get_docs
    events = await _collect_events(chain.astream_events({}, version="v1"))
    _assert_events_equal_allow_superset_metadata(
        events,
        [
            {
                "event": "on_chain_start",
                "run_id": "",
                "parent_ids": [],
                "name": "RunnableSequence",
                "tags": [],
                "metadata": {},
                "data": {"input": {}},
            },
            {
                "event": "on_chain_start",
                "name": "foo",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:1"],
                "metadata": {},
                "data": {},
            },
            {
                "event": "on_chain_stream",
                "name": "foo",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:1"],
                "metadata": {},
                "data": {"chunk": {"x": 5}},
            },
            {
                "event": "on_chain_end",
                "name": "foo",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:1"],
                "metadata": {},
                "data": {"input": {}, "output": {"x": 5}},
            },
            {
                "event": "on_tool_start",
                "name": "get_docs",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:2"],
                "metadata": {},
                "data": {"input": {"x": 5}},
            },
            {
                "event": "on_tool_end",
                "name": "get_docs",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:2"],
                "metadata": {},
                "data": {"input": {"x": 5}, "output": [Document(page_content="hello")]},
            },
            {
                "event": "on_chain_stream",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
                "metadata": {},
                "name": "RunnableSequence",
                "data": {"chunk": [Document(page_content="hello")]},
            },
            {
                "event": "on_chain_end",
                "name": "RunnableSequence",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
                "metadata": {},
                "data": {"output": [Document(page_content="hello")]},
            },
        ],
    )


async def test_event_stream_with_single_lambda() -> None:
    """Test the event stream with a tool."""

    def reverse(s: str) -> str:
        """Reverse a string."""
        return s[::-1]

    chain = RunnableLambda(func=reverse)

    events = await _collect_events(chain.astream_events("hello", version="v1"))
    _assert_events_equal_allow_superset_metadata(
        events,
        [
            {
                "data": {"input": "hello"},
                "event": "on_chain_start",
                "metadata": {},
                "name": "reverse",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {"chunk": "olleh"},
                "event": "on_chain_stream",
                "metadata": {},
                "name": "reverse",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {"output": "olleh"},
                "event": "on_chain_end",
                "metadata": {},
                "name": "reverse",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
        ],
    )


async def test_event_stream_with_triple_lambda() -> None:
    def reverse(s: str) -> str:
        """Reverse a string."""
        return s[::-1]

    r = RunnableLambda(func=reverse)

    chain = (
        r.with_config({"run_name": "1"})
        | r.with_config({"run_name": "2"})
        | r.with_config({"run_name": "3"})
    )
    events = await _collect_events(chain.astream_events("hello", version="v1"))
    _assert_events_equal_allow_superset_metadata(
        events,
        [
            {
                "data": {"input": "hello"},
                "event": "on_chain_start",
                "metadata": {},
                "name": "RunnableSequence",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {},
                "event": "on_chain_start",
                "metadata": {},
                "name": "1",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:1"],
            },
            {
                "data": {"chunk": "olleh"},
                "event": "on_chain_stream",
                "metadata": {},
                "name": "1",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:1"],
            },
            {
                "data": {},
                "event": "on_chain_start",
                "metadata": {},
                "name": "2",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:2"],
            },
            {
                "data": {"input": "hello", "output": "olleh"},
                "event": "on_chain_end",
                "metadata": {},
                "name": "1",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:1"],
            },
            {
                "data": {"chunk": "hello"},
                "event": "on_chain_stream",
                "metadata": {},
                "name": "2",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:2"],
            },
            {
                "data": {},
                "event": "on_chain_start",
                "metadata": {},
                "name": "3",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:3"],
            },
            {
                "data": {"input": "olleh", "output": "hello"},
                "event": "on_chain_end",
                "metadata": {},
                "name": "2",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:2"],
            },
            {
                "data": {"chunk": "olleh"},
                "event": "on_chain_stream",
                "metadata": {},
                "name": "3",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:3"],
            },
            {
                "data": {"chunk": "olleh"},
                "event": "on_chain_stream",
                "metadata": {},
                "name": "RunnableSequence",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {"input": "hello", "output": "olleh"},
                "event": "on_chain_end",
                "metadata": {},
                "name": "3",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:3"],
            },
            {
                "data": {"output": "olleh"},
                "event": "on_chain_end",
                "metadata": {},
                "name": "RunnableSequence",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
        ],
    )


async def test_event_stream_with_triple_lambda_test_filtering() -> None:
    """Test filtering based on tags / names."""

    def reverse(s: str) -> str:
        """Reverse a string."""
        return s[::-1]

    r = RunnableLambda(func=reverse)

    chain = (
        r.with_config({"run_name": "1"})
        | r.with_config({"run_name": "2", "tags": ["my_tag"]})
        | r.with_config({"run_name": "3", "tags": ["my_tag"]})
    )
    events = await _collect_events(
        chain.astream_events("hello", include_names=["1"], version="v1")
    )
    _assert_events_equal_allow_superset_metadata(
        events,
        [
            {
                "data": {},
                "event": "on_chain_start",
                "metadata": {},
                "name": "1",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:1"],
            },
            {
                "data": {"chunk": "olleh"},
                "event": "on_chain_stream",
                "metadata": {},
                "name": "1",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:1"],
            },
            {
                "data": {"input": "hello", "output": "olleh"},
                "event": "on_chain_end",
                "metadata": {},
                "name": "1",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:1"],
            },
        ],
    )

    events = await _collect_events(
        chain.astream_events(
            "hello", include_tags=["my_tag"], exclude_names=["2"], version="v1"
        )
    )
    _assert_events_equal_allow_superset_metadata(
        events,
        [
            {
                "data": {},
                "event": "on_chain_start",
                "metadata": {},
                "name": "3",
                "run_id": "",
                "parent_ids": [],
                "tags": ["my_tag", "seq:step:3"],
            },
            {
                "data": {"chunk": "olleh"},
                "event": "on_chain_stream",
                "metadata": {},
                "name": "3",
                "run_id": "",
                "parent_ids": [],
                "tags": ["my_tag", "seq:step:3"],
            },
            {
                "data": {"input": "hello", "output": "olleh"},
                "event": "on_chain_end",
                "metadata": {},
                "name": "3",
                "run_id": "",
                "parent_ids": [],
                "tags": ["my_tag", "seq:step:3"],
            },
        ],
    )


async def test_event_stream_with_lambdas_from_lambda() -> None:
    as_lambdas = RunnableLambda(lambda _: {"answer": "goodbye"}).with_config(
        {"run_name": "my_lambda"}
    )
    events = await _collect_events(
        as_lambdas.astream_events({"question": "hello"}, version="v1")
    )
    _assert_events_equal_allow_superset_metadata(
        events,
        [
            {
                "data": {"input": {"question": "hello"}},
                "event": "on_chain_start",
                "metadata": {},
                "name": "my_lambda",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {"chunk": {"answer": "goodbye"}},
                "event": "on_chain_stream",
                "metadata": {},
                "name": "my_lambda",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {"output": {"answer": "goodbye"}},
                "event": "on_chain_end",
                "metadata": {},
                "name": "my_lambda",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
        ],
    )


async def test_astream_events_from_model() -> None:
    """Test the output of a model."""
    infinite_cycle = cycle([AIMessage(content="hello world!")])
    # When streaming GenericFakeChatModel breaks AIMessage into chunks based on spaces
    model = (
        GenericFakeChatModel(messages=infinite_cycle)
        .with_config(
            {
                "metadata": {"a": "b"},
                "tags": ["my_model"],
                "run_name": "my_model",
            }
        )
        .bind(stop="<stop_token>")
    )
    events = await _collect_events(model.astream_events("hello", version="v1"))
    _assert_events_equal_allow_superset_metadata(
        events,
        [
            {
                "data": {"input": "hello"},
                "event": "on_chat_model_start",
                "metadata": {"a": "b"},
                "name": "my_model",
                "run_id": "",
                "parent_ids": [],
                "tags": ["my_model"],
            },
            {
                "data": {
                    "chunk": _any_id_ai_message_chunk(
                        content="hello",
                    )
                },
                "event": "on_chat_model_stream",
                "metadata": {"a": "b"},
                "name": "my_model",
                "run_id": "",
                "parent_ids": [],
                "tags": ["my_model"],
            },
            {
                "data": {"chunk": _any_id_ai_message_chunk(content=" ")},
                "event": "on_chat_model_stream",
                "metadata": {"a": "b"},
                "name": "my_model",
                "run_id": "",
                "parent_ids": [],
                "tags": ["my_model"],
            },
            {
                "data": {
                    "chunk": _any_id_ai_message_chunk(
                        content="world!", chunk_position="last"
                    )
                },
                "event": "on_chat_model_stream",
                "metadata": {"a": "b"},
                "name": "my_model",
                "run_id": "",
                "parent_ids": [],
                "tags": ["my_model"],
            },
            {
                "data": {
                    "output": _any_id_ai_message_chunk(
                        content="hello world!", chunk_position="last"
                    )
                },
                "event": "on_chat_model_end",
                "metadata": {"a": "b"},
                "name": "my_model",
                "run_id": "",
                "parent_ids": [],
                "tags": ["my_model"],
            },
        ],
    )

    @RunnableLambda
    def i_dont_stream(value: Any, config: RunnableConfig) -> Any:
        if sys.version_info >= (3, 11):
            return model.invoke(value)
        return model.invoke(value, config)

    events = await _collect_events(i_dont_stream.astream_events("hello", version="v1"))
    _assert_events_equal_allow_superset_metadata(
        events,
        [
            {
                "data": {"input": "hello"},
                "event": "on_chain_start",
                "metadata": {},
                "name": "i_dont_stream",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {"input": {"messages": [[HumanMessage(content="hello")]]}},
                "event": "on_chat_model_start",
                "metadata": {
                    "a": "b",
                    "ls_model_type": "chat",
                    "ls_stop": "<stop_token>",
                },
                "name": "my_model",
                "run_id": "",
                "parent_ids": [],
                "tags": ["my_model"],
            },
            {
                "data": {
                    "chunk": _any_id_ai_message_chunk(
                        content="hello",
                    )
                },
                "event": "on_chat_model_stream",
                "metadata": {
                    "a": "b",
                    "ls_model_type": "chat",
                    "ls_stop": "<stop_token>",
                },
                "name": "my_model",
                "run_id": "",
                "parent_ids": [],
                "tags": ["my_model"],
            },
            {
                "data": {"chunk": _any_id_ai_message_chunk(content=" ")},
                "event": "on_chat_model_stream",
                "metadata": {
                    "a": "b",
                    "ls_model_type": "chat",
                    "ls_stop": "<stop_token>",
                },
                "name": "my_model",
                "run_id": "",
                "parent_ids": [],
                "tags": ["my_model"],
            },
            {
                "data": {
                    "chunk": _any_id_ai_message_chunk(
                        content="world!", chunk_position="last"
                    )
                },
                "event": "on_chat_model_stream",
                "metadata": {
                    "a": "b",
                    "ls_model_type": "chat",
                    "ls_stop": "<stop_token>",
                },
                "name": "my_model",
                "run_id": "",
                "parent_ids": [],
                "tags": ["my_model"],
            },
            {
                "data": {
                    "input": {"messages": [[HumanMessage(content="hello")]]},
                    "output": {
                        "generations": [
                            [
                                {
                                    "generation_info": None,
                                    "message": _any_id_ai_message(
                                        content="hello world!"
                                    ),
                                    "text": "hello world!",
                                    "type": "ChatGeneration",
                                }
                            ]
                        ],
                        "llm_output": None,
                        "run": None,
                        "type": "LLMResult",
                    },
                },
                "event": "on_chat_model_end",
                "metadata": {
                    "a": "b",
                    "ls_model_type": "chat",
                    "ls_stop": "<stop_token>",
                },
                "name": "my_model",
                "run_id": "",
                "parent_ids": [],
                "tags": ["my_model"],
            },
            {
                "data": {"chunk": _any_id_ai_message(content="hello world!")},
                "event": "on_chain_stream",
                "metadata": {},
                "name": "i_dont_stream",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {"output": _any_id_ai_message(content="hello world!")},
                "event": "on_chain_end",
                "metadata": {},
                "name": "i_dont_stream",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
        ],
    )

    @RunnableLambda
    async def ai_dont_stream(value: Any, config: RunnableConfig) -> Any:
        if sys.version_info >= (3, 11):
            return await model.ainvoke(value)
        return await model.ainvoke(value, config)

    events = await _collect_events(ai_dont_stream.astream_events("hello", version="v1"))
    _assert_events_equal_allow_superset_metadata(
        events,
        [
            {
                "data": {"input": "hello"},
                "event": "on_chain_start",
                "metadata": {},
                "name": "ai_dont_stream",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {"input": {"messages": [[HumanMessage(content="hello")]]}},
                "event": "on_chat_model_start",
                "metadata": {
                    "a": "b",
                    "ls_model_type": "chat",
                    "ls_stop": "<stop_token>",
                },
                "name": "my_model",
                "run_id": "",
                "parent_ids": [],
                "tags": ["my_model"],
            },
            {
                "data": {
                    "chunk": _any_id_ai_message_chunk(
                        content="hello",
                    )
                },
                "event": "on_chat_model_stream",
                "metadata": {
                    "a": "b",
                    "ls_model_type": "chat",
                    "ls_stop": "<stop_token>",
                },
                "name": "my_model",
                "run_id": "",
                "parent_ids": [],
                "tags": ["my_model"],
            },
            {
                "data": {"chunk": _any_id_ai_message_chunk(content=" ")},
                "event": "on_chat_model_stream",
                "metadata": {
                    "a": "b",
                    "ls_model_type": "chat",
                    "ls_stop": "<stop_token>",
                },
                "name": "my_model",
                "run_id": "",
                "parent_ids": [],
                "tags": ["my_model"],
            },
            {
                "data": {
                    "chunk": _any_id_ai_message_chunk(
                        content="world!", chunk_position="last"
                    )
                },
                "event": "on_chat_model_stream",
                "metadata": {
                    "a": "b",
                    "ls_model_type": "chat",
                    "ls_stop": "<stop_token>",
                },
                "name": "my_model",
                "run_id": "",
                "parent_ids": [],
                "tags": ["my_model"],
            },
            {
                "data": {
                    "input": {"messages": [[HumanMessage(content="hello")]]},
                    "output": {
                        "generations": [
                            [
                                {
                                    "generation_info": None,
                                    "message": _any_id_ai_message(
                                        content="hello world!"
                                    ),
                                    "text": "hello world!",
                                    "type": "ChatGeneration",
                                }
                            ]
                        ],
                        "llm_output": None,
                        "run": None,
                        "type": "LLMResult",
                    },
                },
                "event": "on_chat_model_end",
                "metadata": {
                    "a": "b",
                    "ls_model_type": "chat",
                    "ls_stop": "<stop_token>",
                },
                "name": "my_model",
                "run_id": "",
                "parent_ids": [],
                "tags": ["my_model"],
            },
            {
                "data": {"chunk": _any_id_ai_message(content="hello world!")},
                "event": "on_chain_stream",
                "metadata": {},
                "name": "ai_dont_stream",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {"output": _any_id_ai_message(content="hello world!")},
                "event": "on_chain_end",
                "metadata": {},
                "name": "ai_dont_stream",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
        ],
    )


async def test_event_stream_with_simple_chain() -> None:
    """Test as event stream."""
    template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are Cat Agent 007"),
            ("human", "{question}"),
        ]
    ).with_config({"run_name": "my_template", "tags": ["my_template"]})

    infinite_cycle = cycle(
        [
            AIMessage(content="hello world!", id="ai1"),
            AIMessage(content="goodbye world!", id="ai2"),
        ]
    )
    # When streaming GenericFakeChatModel breaks AIMessage into chunks based on spaces
    model = (
        GenericFakeChatModel(messages=infinite_cycle)
        .with_config(
            {
                "metadata": {"a": "b"},
                "tags": ["my_model"],
                "run_name": "my_model",
            }
        )
        .bind(stop="<stop_token>")
    )

    chain = (template | model).with_config(
        {
            "metadata": {"foo": "bar"},
            "tags": ["my_chain"],
            "run_name": "my_chain",
        }
    )

    events = await _collect_events(
        chain.astream_events({"question": "hello"}, version="v1")
    )
    _assert_events_equal_allow_superset_metadata(
        events,
        [
            {
                "data": {"input": {"question": "hello"}},
                "event": "on_chain_start",
                "metadata": {"foo": "bar"},
                "name": "my_chain",
                "run_id": "",
                "parent_ids": [],
                "tags": ["my_chain"],
            },
            {
                "data": {"input": {"question": "hello"}},
                "event": "on_prompt_start",
                "metadata": {"foo": "bar"},
                "name": "my_template",
                "run_id": "",
                "parent_ids": [],
                "tags": ["my_chain", "my_template", "seq:step:1"],
            },
            {
                "data": {
                    "input": {"question": "hello"},
                    "output": ChatPromptValue(
                        messages=[
                            SystemMessage(content="You are Cat Agent 007"),
                            HumanMessage(content="hello"),
                        ]
                    ),
                },
                "event": "on_prompt_end",
                "metadata": {"foo": "bar"},
                "name": "my_template",
                "run_id": "",
                "parent_ids": [],
                "tags": ["my_chain", "my_template", "seq:step:1"],
            },
            {
                "data": {
                    "input": {
                        "messages": [
                            [
                                SystemMessage(content="You are Cat Agent 007"),
                                HumanMessage(content="hello"),
                            ]
                        ]
                    }
                },
                "event": "on_chat_model_start",
                "metadata": {
                    "a": "b",
                    "foo": "bar",
                    "ls_model_type": "chat",
                    "ls_stop": "<stop_token>",
                },
                "name": "my_model",
                "run_id": "",
                "parent_ids": [],
                "tags": ["my_chain", "my_model", "seq:step:2"],
            },
            {
                "data": {
                    "chunk": AIMessageChunk(
                        content="hello",
                        id="ai1",
                    )
                },
                "event": "on_chat_model_stream",
                "metadata": {
                    "a": "b",
                    "foo": "bar",
                    "ls_model_type": "chat",
                    "ls_stop": "<stop_token>",
                },
                "name": "my_model",
                "run_id": "",
                "parent_ids": [],
                "tags": ["my_chain", "my_model", "seq:step:2"],
            },
            {
                "data": {
                    "chunk": AIMessageChunk(
                        content="hello",
                        id="ai1",
                    )
                },
                "event": "on_chain_stream",
                "metadata": {"foo": "bar"},
                "name": "my_chain",
                "run_id": "",
                "parent_ids": [],
                "tags": ["my_chain"],
            },
            {
                "data": {"chunk": AIMessageChunk(content=" ", id="ai1")},
                "event": "on_chat_model_stream",
                "metadata": {
                    "a": "b",
                    "foo": "bar",
                    "ls_model_type": "chat",
                    "ls_stop": "<stop_token>",
                },
                "name": "my_model",
                "run_id": "",
                "parent_ids": [],
                "tags": ["my_chain", "my_model", "seq:step:2"],
            },
            {
                "data": {"chunk": AIMessageChunk(content=" ", id="ai1")},
                "event": "on_chain_stream",
                "metadata": {"foo": "bar"},
                "name": "my_chain",
                "run_id": "",
                "parent_ids": [],
                "tags": ["my_chain"],
            },
            {
                "data": {
                    "chunk": AIMessageChunk(
                        content="world!", id="ai1", chunk_position="last"
                    )
                },
                "event": "on_chat_model_stream",
                "metadata": {
                    "a": "b",
                    "foo": "bar",
                    "ls_model_type": "chat",
                    "ls_stop": "<stop_token>",
                },
                "name": "my_model",
                "run_id": "",
                "parent_ids": [],
                "tags": ["my_chain", "my_model", "seq:step:2"],
            },
            {
                "data": {
                    "chunk": AIMessageChunk(
                        content="world!", id="ai1", chunk_position="last"
                    )
                },
                "event": "on_chain_stream",
                "metadata": {"foo": "bar"},
                "name": "my_chain",
                "run_id": "",
                "parent_ids": [],
                "tags": ["my_chain"],
            },
            {
                "data": {
                    "input": {
                        "messages": [
                            [
                                SystemMessage(content="You are Cat Agent 007"),
                                HumanMessage(content="hello"),
                            ]
                        ]
                    },
                    "output": {
                        "generations": [
                            [
                                {
                                    "generation_info": None,
                                    "message": AIMessageChunk(
                                        content="hello world!",
                                        id="ai1",
                                        chunk_position="last",
                                    ),
                                    "text": "hello world!",
                                    "type": "ChatGenerationChunk",
                                }
                            ]
                        ],
                        "llm_output": None,
                        "run": None,
                        "type": "LLMResult",
                    },
                },
                "event": "on_chat_model_end",
                "metadata": {
                    "a": "b",
                    "foo": "bar",
                    "ls_model_type": "chat",
                    "ls_stop": "<stop_token>",
                },
                "name": "my_model",
                "run_id": "",
                "parent_ids": [],
                "tags": ["my_chain", "my_model", "seq:step:2"],
            },
            {
                "data": {
                    "output": AIMessageChunk(
                        content="hello world!", id="ai1", chunk_position="last"
                    )
                },
                "event": "on_chain_end",
                "metadata": {"foo": "bar"},
                "name": "my_chain",
                "run_id": "",
                "parent_ids": [],
                "tags": ["my_chain"],
            },
        ],
    )


async def test_event_streaming_with_tools() -> None:
    """Test streaming events with different tool definitions."""

    @tool
    def parameterless() -> str:
        """A tool that does nothing."""
        return "hello"

    @tool
    def with_callbacks(callbacks: Callbacks) -> str:  # noqa: ARG001
        """A tool that does nothing."""
        return "world"

    @tool
    def with_parameters(x: int, y: str) -> dict:
        """A tool that does nothing."""
        return {"x": x, "y": y}

    @tool
    def with_parameters_and_callbacks(x: int, y: str, callbacks: Callbacks) -> dict:  # noqa: ARG001
        """A tool that does nothing."""
        return {"x": x, "y": y}

    # type ignores below because the tools don't appear to be runnables to type checkers
    # we can remove as soon as that's fixed
    events = await _collect_events(parameterless.astream_events({}, version="v1"))
    _assert_events_equal_allow_superset_metadata(
        events,
        [
            {
                "data": {"input": {}},
                "event": "on_tool_start",
                "metadata": {},
                "name": "parameterless",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {"chunk": "hello"},
                "event": "on_tool_stream",
                "metadata": {},
                "name": "parameterless",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {"output": "hello"},
                "event": "on_tool_end",
                "metadata": {},
                "name": "parameterless",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
        ],
    )

    events = await _collect_events(with_callbacks.astream_events({}, version="v1"))
    _assert_events_equal_allow_superset_metadata(
        events,
        [
            {
                "data": {"input": {}},
                "event": "on_tool_start",
                "metadata": {},
                "name": "with_callbacks",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {"chunk": "world"},
                "event": "on_tool_stream",
                "metadata": {},
                "name": "with_callbacks",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {"output": "world"},
                "event": "on_tool_end",
                "metadata": {},
                "name": "with_callbacks",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
        ],
    )
    events = await _collect_events(
        with_parameters.astream_events({"x": 1, "y": "2"}, version="v1")
    )
    _assert_events_equal_allow_superset_metadata(
        events,
        [
            {
                "data": {"input": {"x": 1, "y": "2"}},
                "event": "on_tool_start",
                "metadata": {},
                "name": "with_parameters",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {"chunk": {"x": 1, "y": "2"}},
                "event": "on_tool_stream",
                "metadata": {},
                "name": "with_parameters",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {"output": {"x": 1, "y": "2"}},
                "event": "on_tool_end",
                "metadata": {},
                "name": "with_parameters",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
        ],
    )

    events = await _collect_events(
        with_parameters_and_callbacks.astream_events({"x": 1, "y": "2"}, version="v1")
    )
    _assert_events_equal_allow_superset_metadata(
        events,
        [
            {
                "data": {"input": {"x": 1, "y": "2"}},
                "event": "on_tool_start",
                "metadata": {},
                "name": "with_parameters_and_callbacks",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {"chunk": {"x": 1, "y": "2"}},
                "event": "on_tool_stream",
                "metadata": {},
                "name": "with_parameters_and_callbacks",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {"output": {"x": 1, "y": "2"}},
                "event": "on_tool_end",
                "metadata": {},
                "name": "with_parameters_and_callbacks",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
        ],
    )


class HardCodedRetriever(BaseRetriever):
    documents: list[Document]

    @override
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.documents


async def test_event_stream_with_retriever() -> None:
    """Test the event stream with a retriever."""
    retriever = HardCodedRetriever(
        documents=[
            Document(
                page_content="hello world!",
                metadata={"foo": "bar"},
            ),
            Document(
                page_content="goodbye world!",
                metadata={"food": "spare"},
            ),
        ]
    )
    events = await _collect_events(
        retriever.astream_events({"query": "hello"}, version="v1")
    )
    _assert_events_equal_allow_superset_metadata(
        events,
        [
            {
                "data": {
                    "input": {"query": "hello"},
                },
                "event": "on_retriever_start",
                "metadata": {},
                "name": "HardCodedRetriever",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {
                    "chunk": [
                        Document(page_content="hello world!", metadata={"foo": "bar"}),
                        Document(
                            page_content="goodbye world!", metadata={"food": "spare"}
                        ),
                    ]
                },
                "event": "on_retriever_stream",
                "metadata": {},
                "name": "HardCodedRetriever",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {
                    "output": [
                        Document(page_content="hello world!", metadata={"foo": "bar"}),
                        Document(
                            page_content="goodbye world!", metadata={"food": "spare"}
                        ),
                    ],
                },
                "event": "on_retriever_end",
                "metadata": {},
                "name": "HardCodedRetriever",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
        ],
    )


async def test_event_stream_with_retriever_and_formatter() -> None:
    """Test the event stream with a retriever."""
    retriever = HardCodedRetriever(
        documents=[
            Document(
                page_content="hello world!",
                metadata={"foo": "bar"},
            ),
            Document(
                page_content="goodbye world!",
                metadata={"food": "spare"},
            ),
        ]
    )

    def format_docs(docs: list[Document]) -> str:
        """Format the docs."""
        return ", ".join([doc.page_content for doc in docs])

    chain = retriever | format_docs
    events = await _collect_events(chain.astream_events("hello", version="v1"))
    _assert_events_equal_allow_superset_metadata(
        events,
        [
            {
                "data": {"input": "hello"},
                "event": "on_chain_start",
                "metadata": {},
                "name": "RunnableSequence",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {"input": {"query": "hello"}},
                "event": "on_retriever_start",
                "metadata": {},
                "name": "HardCodedRetriever",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:1"],
            },
            {
                "data": {
                    "input": {"query": "hello"},
                    "output": {
                        "documents": [
                            Document(
                                page_content="hello world!", metadata={"foo": "bar"}
                            ),
                            Document(
                                page_content="goodbye world!",
                                metadata={"food": "spare"},
                            ),
                        ]
                    },
                },
                "event": "on_retriever_end",
                "metadata": {},
                "name": "HardCodedRetriever",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:1"],
            },
            {
                "data": {},
                "event": "on_chain_start",
                "metadata": {},
                "name": "format_docs",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:2"],
            },
            {
                "data": {"chunk": "hello world!, goodbye world!"},
                "event": "on_chain_stream",
                "metadata": {},
                "name": "format_docs",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:2"],
            },
            {
                "data": {"chunk": "hello world!, goodbye world!"},
                "event": "on_chain_stream",
                "metadata": {},
                "name": "RunnableSequence",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {
                    "input": [
                        Document(page_content="hello world!", metadata={"foo": "bar"}),
                        Document(
                            page_content="goodbye world!", metadata={"food": "spare"}
                        ),
                    ],
                    "output": "hello world!, goodbye world!",
                },
                "event": "on_chain_end",
                "metadata": {},
                "name": "format_docs",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:2"],
            },
            {
                "data": {"output": "hello world!, goodbye world!"},
                "event": "on_chain_end",
                "metadata": {},
                "name": "RunnableSequence",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
        ],
    )


async def test_event_stream_on_chain_with_tool() -> None:
    """Test the event stream with a tool."""

    @tool
    def concat(a: str, b: str) -> str:
        """A tool that does nothing."""
        return a + b

    def reverse(s: str) -> str:
        """Reverse a string."""
        return s[::-1]

    # For whatever reason type annotations fail here because reverse
    # does not appear to be a runnable
    chain = concat | reverse

    events = await _collect_events(
        chain.astream_events({"a": "hello", "b": "world"}, version="v1")
    )
    _assert_events_equal_allow_superset_metadata(
        events,
        [
            {
                "data": {"input": {"a": "hello", "b": "world"}},
                "event": "on_chain_start",
                "metadata": {},
                "name": "RunnableSequence",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {"input": {"a": "hello", "b": "world"}},
                "event": "on_tool_start",
                "metadata": {},
                "name": "concat",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:1"],
            },
            {
                "data": {"input": {"a": "hello", "b": "world"}, "output": "helloworld"},
                "event": "on_tool_end",
                "metadata": {},
                "name": "concat",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:1"],
            },
            {
                "data": {},
                "event": "on_chain_start",
                "metadata": {},
                "name": "reverse",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:2"],
            },
            {
                "data": {"chunk": "dlrowolleh"},
                "event": "on_chain_stream",
                "metadata": {},
                "name": "reverse",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:2"],
            },
            {
                "data": {"chunk": "dlrowolleh"},
                "event": "on_chain_stream",
                "metadata": {},
                "name": "RunnableSequence",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {"input": "helloworld", "output": "dlrowolleh"},
                "event": "on_chain_end",
                "metadata": {},
                "name": "reverse",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:2"],
            },
            {
                "data": {"output": "dlrowolleh"},
                "event": "on_chain_end",
                "metadata": {},
                "name": "RunnableSequence",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
        ],
    )


@pytest.mark.xfail(reason="Fix order of callback invocations in RunnableSequence")
async def test_chain_ordering() -> None:
    """Test the event stream with a tool."""

    def foo(a: str) -> str:
        return a

    def bar(a: str) -> str:
        return a

    chain = RunnableLambda(foo) | RunnableLambda(bar)
    iterable = chain.astream_events("q", version="v1")

    events = []

    try:
        for _ in range(10):
            next_chunk = await anext(iterable)
            events.append(next_chunk)
    except Exception:
        pass

    events = _with_nulled_run_id(events)
    for event in events:
        event["tags"] = sorted(event["tags"])

    _assert_events_equal_allow_superset_metadata(
        events,
        [
            {
                "data": {"input": "q"},
                "event": "on_chain_start",
                "metadata": {},
                "name": "RunnableSequence",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {},
                "event": "on_chain_start",
                "metadata": {},
                "name": "foo",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:1"],
            },
            {
                "data": {"chunk": "q"},
                "event": "on_chain_stream",
                "metadata": {},
                "name": "foo",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:1"],
            },
            {
                "data": {"input": "q", "output": "q"},
                "event": "on_chain_end",
                "metadata": {},
                "name": "foo",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:1"],
            },
            {
                "data": {},
                "event": "on_chain_start",
                "metadata": {},
                "name": "bar",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:2"],
            },
            {
                "data": {"chunk": "q"},
                "event": "on_chain_stream",
                "metadata": {},
                "name": "bar",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:2"],
            },
            {
                "data": {"chunk": "q"},
                "event": "on_chain_stream",
                "metadata": {},
                "name": "RunnableSequence",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {"input": "q", "output": "q"},
                "event": "on_chain_end",
                "metadata": {},
                "name": "bar",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:2"],
            },
            {
                "data": {"output": "q"},
                "event": "on_chain_end",
                "metadata": {},
                "name": "RunnableSequence",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
        ],
    )


async def test_event_stream_with_retry() -> None:
    """Test the event stream with a tool."""

    def success(_: str) -> str:
        return "success"

    def fail(_: str) -> None:
        """Simple func."""
        msg = "fail"
        raise ValueError(msg)

    chain = RunnableLambda(success) | RunnableLambda(fail).with_retry(
        stop_after_attempt=1,
    )
    iterable = chain.astream_events("q", version="v1")

    events = []

    try:
        for _ in range(10):
            next_chunk = await anext(iterable)
            events.append(next_chunk)
    except Exception:
        pass

    events = _with_nulled_run_id(events)
    for event in events:
        event["tags"] = sorted(event["tags"])

    _assert_events_equal_allow_superset_metadata(
        events,
        [
            {
                "data": {"input": "q"},
                "event": "on_chain_start",
                "metadata": {},
                "name": "RunnableSequence",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {},
                "event": "on_chain_start",
                "metadata": {},
                "name": "success",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:1"],
            },
            {
                "data": {"chunk": "success"},
                "event": "on_chain_stream",
                "metadata": {},
                "name": "success",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:1"],
            },
            {
                "data": {},
                "event": "on_chain_start",
                "metadata": {},
                "name": "fail",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:2"],
            },
            {
                "data": {"input": "q", "output": "success"},
                "event": "on_chain_end",
                "metadata": {},
                "name": "success",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:1"],
            },
            {
                "data": {"input": "success", "output": None},
                "event": "on_chain_end",
                "metadata": {},
                "name": "fail",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:2"],
            },
        ],
    )


async def test_with_llm() -> None:
    """Test with regular llm."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are Cat Agent 007"),
            ("human", "{question}"),
        ]
    ).with_config({"run_name": "my_template", "tags": ["my_template"]})
    llm = FakeStreamingListLLM(responses=["abc"])

    chain = prompt | llm
    events = await _collect_events(
        chain.astream_events({"question": "hello"}, version="v1")
    )
    _assert_events_equal_allow_superset_metadata(
        events,
        [
            {
                "data": {"input": {"question": "hello"}},
                "event": "on_chain_start",
                "metadata": {},
                "name": "RunnableSequence",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {"input": {"question": "hello"}},
                "event": "on_prompt_start",
                "metadata": {},
                "name": "my_template",
                "run_id": "",
                "parent_ids": [],
                "tags": ["my_template", "seq:step:1"],
            },
            {
                "data": {
                    "input": {"question": "hello"},
                    "output": ChatPromptValue(
                        messages=[
                            SystemMessage(content="You are Cat Agent 007"),
                            HumanMessage(content="hello"),
                        ]
                    ),
                },
                "event": "on_prompt_end",
                "metadata": {},
                "name": "my_template",
                "run_id": "",
                "parent_ids": [],
                "tags": ["my_template", "seq:step:1"],
            },
            {
                "data": {
                    "input": {
                        "prompts": ["System: You are Cat Agent 007\nHuman: hello"]
                    }
                },
                "event": "on_llm_start",
                "metadata": {},
                "name": "FakeStreamingListLLM",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:2"],
            },
            {
                "data": {
                    "input": {
                        "prompts": ["System: You are Cat Agent 007\nHuman: hello"]
                    },
                    "output": {
                        "generations": [
                            [
                                {
                                    "generation_info": None,
                                    "text": "abc",
                                    "type": "Generation",
                                }
                            ]
                        ],
                        "llm_output": None,
                        "run": None,
                        "type": "LLMResult",
                    },
                },
                "event": "on_llm_end",
                "metadata": {},
                "name": "FakeStreamingListLLM",
                "run_id": "",
                "parent_ids": [],
                "tags": ["seq:step:2"],
            },
            {
                "data": {"chunk": "a"},
                "event": "on_chain_stream",
                "metadata": {},
                "name": "RunnableSequence",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {"chunk": "b"},
                "event": "on_chain_stream",
                "metadata": {},
                "name": "RunnableSequence",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {"chunk": "c"},
                "event": "on_chain_stream",
                "metadata": {},
                "name": "RunnableSequence",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {"output": "abc"},
                "event": "on_chain_end",
                "metadata": {},
                "name": "RunnableSequence",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
        ],
    )


async def test_runnable_each() -> None:
    """Test runnable each astream_events."""

    async def add_one(x: int) -> int:
        return x + 1

    add_one_map = RunnableLambda(add_one).map()  # type: ignore[arg-type,var-annotated]
    assert await add_one_map.ainvoke([1, 2, 3]) == [2, 3, 4]

    with pytest.raises(NotImplementedError):
        _ = [_ async for _ in add_one_map.astream_events([1, 2, 3], version="v1")]


async def test_events_astream_config() -> None:
    """Test that astream events support accepting config."""
    infinite_cycle = cycle([AIMessage(content="hello world!", id="ai1")])
    good_world_on_repeat = cycle([AIMessage(content="Goodbye world", id="ai2")])
    model = GenericFakeChatModel(messages=infinite_cycle).configurable_fields(
        messages=ConfigurableField(
            id="messages",
            name="Messages",
            description="Messages return by the LLM",
        )
    )

    model_02 = model.with_config({"configurable": {"messages": good_world_on_repeat}})
    assert model_02.invoke("hello") == AIMessage(content="Goodbye world", id="ai2")

    events = await _collect_events(model_02.astream_events("hello", version="v1"))
    _assert_events_equal_allow_superset_metadata(
        events,
        [
            {
                "data": {"input": "hello"},
                "event": "on_chat_model_start",
                "metadata": {},
                "name": "RunnableConfigurableFields",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {
                    "chunk": AIMessageChunk(
                        content="Goodbye",
                        id="ai2",
                    )
                },
                "event": "on_chat_model_stream",
                "metadata": {},
                "name": "RunnableConfigurableFields",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {"chunk": AIMessageChunk(content=" ", id="ai2")},
                "event": "on_chat_model_stream",
                "metadata": {},
                "name": "RunnableConfigurableFields",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {
                    "chunk": AIMessageChunk(
                        content="world", id="ai2", chunk_position="last"
                    )
                },
                "event": "on_chat_model_stream",
                "metadata": {},
                "name": "RunnableConfigurableFields",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
            {
                "data": {
                    "output": AIMessageChunk(
                        content="Goodbye world", id="ai2", chunk_position="last"
                    )
                },
                "event": "on_chat_model_end",
                "metadata": {},
                "name": "RunnableConfigurableFields",
                "run_id": "",
                "parent_ids": [],
                "tags": [],
            },
        ],
    )


async def test_runnable_with_message_history() -> None:
    class InMemoryHistory(BaseChatMessageHistory, BaseModel):
        """In memory implementation of chat message history."""

        # Attention: for the tests use an Any type to work-around a pydantic issue
        # where it re-instantiates a list, so mutating the list doesn't end up mutating
        # the content in the store!

        # Using Any type here rather than list[BaseMessage] due to pydantic issue!
        messages: Any

        def add_message(self, message: BaseMessage) -> None:
            """Add a self-created message to the store."""
            self.messages.append(message)

        def clear(self) -> None:
            self.messages = []

    # Here we use a global variable to store the chat message history.
    # This will make it easier to inspect it to see the underlying results.
    store: dict = {}

    def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
        """Get a chat message history."""
        if session_id not in store:
            store[session_id] = []
        return InMemoryHistory(messages=store[session_id])

    infinite_cycle = cycle(
        [
            AIMessage(content="hello", id="ai3"),
            AIMessage(content="world", id="ai4"),
        ]
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a cat"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )
    model = GenericFakeChatModel(messages=infinite_cycle)

    chain = prompt | model
    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history=get_by_session_id,
        input_messages_key="question",
        history_messages_key="history",
    )
    await with_message_history.with_config(
        {"configurable": {"session_id": "session-123"}}
    ).ainvoke({"question": "hello"})

    assert store == {
        "session-123": [
            HumanMessage(content="hello"),
            AIMessage(content="hello", id="ai3"),
        ]
    }

    await asyncio.to_thread(
        with_message_history.with_config(
            {"configurable": {"session_id": "session-123"}}
        ).invoke,
        {"question": "meow"},
    )
    assert store == {
        "session-123": [
            HumanMessage(content="hello"),
            AIMessage(content="hello", id="ai3"),
            HumanMessage(content="meow"),
            AIMessage(content="world", id="ai4"),
        ]
    }


EXPECTED_EVENTS = [
    {
        "data": {"input": 1},
        "event": "on_chain_start",
        "metadata": {},
        "name": "add_one_proxy",
        "run_id": "",
        "parent_ids": [],
        "tags": [],
    },
    {
        "data": {},
        "event": "on_chain_start",
        "metadata": {},
        "name": "add_one",
        "run_id": "",
        "parent_ids": [],
        "tags": [],
    },
    {
        "data": {"chunk": 2},
        "event": "on_chain_stream",
        "metadata": {},
        "name": "add_one",
        "run_id": "",
        "parent_ids": [],
        "tags": [],
    },
    {
        "data": {"input": 1, "output": 2},
        "event": "on_chain_end",
        "metadata": {},
        "name": "add_one",
        "run_id": "",
        "parent_ids": [],
        "tags": [],
    },
    {
        "data": {"chunk": 2},
        "event": "on_chain_stream",
        "metadata": {},
        "name": "add_one_proxy",
        "run_id": "",
        "parent_ids": [],
        "tags": [],
    },
    {
        "data": {"output": 2},
        "event": "on_chain_end",
        "metadata": {},
        "name": "add_one_proxy",
        "run_id": "",
        "parent_ids": [],
        "tags": [],
    },
]


@pytest.mark.xfail(
    reason="This test is failing due to missing functionality."
    "Need to implement logic in _transform_stream_with_config that mimics the async "
    "variant that uses tap_output_iter"
)
async def test_sync_in_async_stream_lambdas() -> None:
    """Test invoking nested runnable lambda."""

    def add_one_(x: int) -> int:
        return x + 1

    add_one = RunnableLambda(add_one_)

    async def add_one_proxy_(x: int, config: RunnableConfig) -> int:
        streaming = add_one.stream(x, config)
        results = list(streaming)
        return results[0]

    add_one_proxy = RunnableLambda(add_one_proxy_)  # type: ignore[arg-type,var-annotated]

    events = await _collect_events(add_one_proxy.astream_events(1, version="v1"))
    _assert_events_equal_allow_superset_metadata(events, EXPECTED_EVENTS)


async def test_async_in_async_stream_lambdas() -> None:
    """Test invoking nested runnable lambda."""

    async def add_one(x: int) -> int:
        return x + 1

    add_one_ = RunnableLambda(add_one)  # type: ignore[arg-type,var-annotated]

    async def add_one_proxy(x: int, config: RunnableConfig) -> int:
        # Use sync streaming
        streaming = add_one_.astream(x, config)
        results = [result async for result in streaming]
        return results[0]

    add_one_proxy_ = RunnableLambda(add_one_proxy)  # type: ignore[arg-type,var-annotated]

    events = await _collect_events(add_one_proxy_.astream_events(1, version="v1"))
    _assert_events_equal_allow_superset_metadata(events, EXPECTED_EVENTS)


@pytest.mark.xfail(
    reason="This test is failing due to missing functionality."
    "Need to implement logic in _transform_stream_with_config that mimics the async "
    "variant that uses tap_output_iter"
)
async def test_sync_in_sync_lambdas() -> None:
    """Test invoking nested runnable lambda."""

    def add_one(x: int) -> int:
        return x + 1

    add_one_ = RunnableLambda(add_one)

    def add_one_proxy(x: int, config: RunnableConfig) -> int:
        # Use sync streaming
        streaming = add_one_.stream(x, config)
        results = list(streaming)
        return results[0]

    add_one_proxy_ = RunnableLambda(add_one_proxy)

    events = await _collect_events(add_one_proxy_.astream_events(1, version="v1"))
    _assert_events_equal_allow_superset_metadata(events, EXPECTED_EVENTS)
