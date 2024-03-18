"""Module that contains tests for runnable.astream_events API."""
from itertools import cycle
from typing import Any, AsyncIterator, Dict, List, Sequence, cast

import pytest

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
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    ConfigurableField,
    Runnable,
    RunnableLambda,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.schema import StreamEvent
from langchain_core.tools import tool


def _with_nulled_run_id(events: Sequence[StreamEvent]) -> List[StreamEvent]:
    """Removes the run ids from events."""
    return cast(List[StreamEvent], [{**event, "run_id": ""} for event in events])


async def _as_async_iterator(iterable: List) -> AsyncIterator:
    """Converts an iterable into an async iterator."""
    for item in iterable:
        yield item


async def _collect_events(events: AsyncIterator[StreamEvent]) -> List[StreamEvent]:
    """Collect the events and remove the run ids."""
    materialized_events = [event async for event in events]
    events_ = _with_nulled_run_id(materialized_events)
    for event in events_:
        event["tags"] = sorted(event["tags"])
    return events_


async def test_event_stream_with_single_lambda() -> None:
    """Test the event stream with a tool."""

    def reverse(s: str) -> str:
        """Reverse a string."""
        return s[::-1]

    chain = RunnableLambda(func=reverse)

    events = await _collect_events(chain.astream_events("hello", version="v1"))
    assert events == [
        {
            "data": {"input": "hello"},
            "event": "on_chain_start",
            "metadata": {},
            "name": "reverse",
            "run_id": "",
            "tags": [],
        },
        {
            "data": {"chunk": "olleh"},
            "event": "on_chain_stream",
            "metadata": {},
            "name": "reverse",
            "run_id": "",
            "tags": [],
        },
        {
            "data": {"output": "olleh"},
            "event": "on_chain_end",
            "metadata": {},
            "name": "reverse",
            "run_id": "",
            "tags": [],
        },
    ]


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
    assert events == [
        {
            "data": {"input": "hello"},
            "event": "on_chain_start",
            "metadata": {},
            "name": "RunnableSequence",
            "run_id": "",
            "tags": [],
        },
        {
            "data": {},
            "event": "on_chain_start",
            "metadata": {},
            "name": "1",
            "run_id": "",
            "tags": ["seq:step:1"],
        },
        {
            "data": {"chunk": "olleh"},
            "event": "on_chain_stream",
            "metadata": {},
            "name": "1",
            "run_id": "",
            "tags": ["seq:step:1"],
        },
        {
            "data": {},
            "event": "on_chain_start",
            "metadata": {},
            "name": "2",
            "run_id": "",
            "tags": ["seq:step:2"],
        },
        {
            "data": {"input": "hello", "output": "olleh"},
            "event": "on_chain_end",
            "metadata": {},
            "name": "1",
            "run_id": "",
            "tags": ["seq:step:1"],
        },
        {
            "data": {"chunk": "hello"},
            "event": "on_chain_stream",
            "metadata": {},
            "name": "2",
            "run_id": "",
            "tags": ["seq:step:2"],
        },
        {
            "data": {},
            "event": "on_chain_start",
            "metadata": {},
            "name": "3",
            "run_id": "",
            "tags": ["seq:step:3"],
        },
        {
            "data": {"input": "olleh", "output": "hello"},
            "event": "on_chain_end",
            "metadata": {},
            "name": "2",
            "run_id": "",
            "tags": ["seq:step:2"],
        },
        {
            "data": {"chunk": "olleh"},
            "event": "on_chain_stream",
            "metadata": {},
            "name": "3",
            "run_id": "",
            "tags": ["seq:step:3"],
        },
        {
            "data": {"chunk": "olleh"},
            "event": "on_chain_stream",
            "metadata": {},
            "name": "RunnableSequence",
            "run_id": "",
            "tags": [],
        },
        {
            "data": {"input": "hello", "output": "olleh"},
            "event": "on_chain_end",
            "metadata": {},
            "name": "3",
            "run_id": "",
            "tags": ["seq:step:3"],
        },
        {
            "data": {"output": "olleh"},
            "event": "on_chain_end",
            "metadata": {},
            "name": "RunnableSequence",
            "run_id": "",
            "tags": [],
        },
    ]


async def test_event_stream_with_triple_lambda_test_filtering() -> None:
    """Test filtering based on tags / names"""

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
    assert events == [
        {
            "data": {},
            "event": "on_chain_start",
            "metadata": {},
            "name": "1",
            "run_id": "",
            "tags": ["seq:step:1"],
        },
        {
            "data": {"chunk": "olleh"},
            "event": "on_chain_stream",
            "metadata": {},
            "name": "1",
            "run_id": "",
            "tags": ["seq:step:1"],
        },
        {
            "data": {"input": "hello", "output": "olleh"},
            "event": "on_chain_end",
            "metadata": {},
            "name": "1",
            "run_id": "",
            "tags": ["seq:step:1"],
        },
    ]

    events = await _collect_events(
        chain.astream_events(
            "hello", include_tags=["my_tag"], exclude_names=["2"], version="v1"
        )
    )
    assert events == [
        {
            "data": {},
            "event": "on_chain_start",
            "metadata": {},
            "name": "3",
            "run_id": "",
            "tags": ["my_tag", "seq:step:3"],
        },
        {
            "data": {"chunk": "olleh"},
            "event": "on_chain_stream",
            "metadata": {},
            "name": "3",
            "run_id": "",
            "tags": ["my_tag", "seq:step:3"],
        },
        {
            "data": {"input": "hello", "output": "olleh"},
            "event": "on_chain_end",
            "metadata": {},
            "name": "3",
            "run_id": "",
            "tags": ["my_tag", "seq:step:3"],
        },
    ]


async def test_event_stream_with_lambdas_from_lambda() -> None:
    as_lambdas = RunnableLambda(lambda x: {"answer": "goodbye"}).with_config(
        {"run_name": "my_lambda"}
    )
    events = await _collect_events(
        as_lambdas.astream_events({"question": "hello"}, version="v1")
    )
    assert events == [
        {
            "data": {"input": {"question": "hello"}},
            "event": "on_chain_start",
            "metadata": {},
            "name": "my_lambda",
            "run_id": "",
            "tags": [],
        },
        {
            "data": {"chunk": {"answer": "goodbye"}},
            "event": "on_chain_stream",
            "metadata": {},
            "name": "my_lambda",
            "run_id": "",
            "tags": [],
        },
        {
            "data": {"output": {"answer": "goodbye"}},
            "event": "on_chain_end",
            "metadata": {},
            "name": "my_lambda",
            "run_id": "",
            "tags": [],
        },
    ]


async def test_event_stream_with_simple_chain() -> None:
    """Test as event stream."""
    template = ChatPromptTemplate.from_messages(
        [("system", "You are Cat Agent 007"), ("human", "{question}")]
    ).with_config({"run_name": "my_template", "tags": ["my_template"]})

    infinite_cycle = cycle(
        [AIMessage(content="hello world!"), AIMessage(content="goodbye world!")]
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
    assert events == [
        {
            "data": {"input": {"question": "hello"}},
            "event": "on_chain_start",
            "metadata": {"foo": "bar"},
            "name": "my_chain",
            "run_id": "",
            "tags": ["my_chain"],
        },
        {
            "data": {"input": {"question": "hello"}},
            "event": "on_prompt_start",
            "metadata": {"foo": "bar"},
            "name": "my_template",
            "run_id": "",
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
            "metadata": {"a": "b", "foo": "bar"},
            "name": "my_model",
            "run_id": "",
            "tags": ["my_chain", "my_model", "seq:step:2"],
        },
        {
            "data": {"chunk": AIMessageChunk(content="hello")},
            "event": "on_chat_model_stream",
            "metadata": {"a": "b", "foo": "bar"},
            "name": "my_model",
            "run_id": "",
            "tags": ["my_chain", "my_model", "seq:step:2"],
        },
        {
            "data": {"chunk": AIMessageChunk(content="hello")},
            "event": "on_chain_stream",
            "metadata": {"foo": "bar"},
            "name": "my_chain",
            "run_id": "",
            "tags": ["my_chain"],
        },
        {
            "data": {"chunk": AIMessageChunk(content=" ")},
            "event": "on_chat_model_stream",
            "metadata": {"a": "b", "foo": "bar"},
            "name": "my_model",
            "run_id": "",
            "tags": ["my_chain", "my_model", "seq:step:2"],
        },
        {
            "data": {"chunk": AIMessageChunk(content=" ")},
            "event": "on_chain_stream",
            "metadata": {"foo": "bar"},
            "name": "my_chain",
            "run_id": "",
            "tags": ["my_chain"],
        },
        {
            "data": {"chunk": AIMessageChunk(content="world!")},
            "event": "on_chat_model_stream",
            "metadata": {"a": "b", "foo": "bar"},
            "name": "my_model",
            "run_id": "",
            "tags": ["my_chain", "my_model", "seq:step:2"],
        },
        {
            "data": {"chunk": AIMessageChunk(content="world!")},
            "event": "on_chain_stream",
            "metadata": {"foo": "bar"},
            "name": "my_chain",
            "run_id": "",
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
                                "message": AIMessageChunk(content="hello world!"),
                                "text": "hello world!",
                                "type": "ChatGenerationChunk",
                            }
                        ]
                    ],
                    "llm_output": None,
                    "run": None,
                },
            },
            "event": "on_chat_model_end",
            "metadata": {"a": "b", "foo": "bar"},
            "name": "my_model",
            "run_id": "",
            "tags": ["my_chain", "my_model", "seq:step:2"],
        },
        {
            "data": {"output": AIMessageChunk(content="hello world!")},
            "event": "on_chain_end",
            "metadata": {"foo": "bar"},
            "name": "my_chain",
            "run_id": "",
            "tags": ["my_chain"],
        },
    ]


async def test_event_streaming_with_tools() -> None:
    """Test streaming events with different tool definitions."""

    @tool
    def parameterless() -> str:
        """A tool that does nothing."""
        return "hello"

    @tool
    def with_callbacks(callbacks: Callbacks) -> str:
        """A tool that does nothing."""
        return "world"

    @tool
    def with_parameters(x: int, y: str) -> dict:
        """A tool that does nothing."""
        return {"x": x, "y": y}

    @tool
    def with_parameters_and_callbacks(x: int, y: str, callbacks: Callbacks) -> dict:
        """A tool that does nothing."""
        return {"x": x, "y": y}

    # type ignores below because the tools don't appear to be runnables to type checkers
    # we can remove as soon as that's fixed
    events = await _collect_events(parameterless.astream_events({}, version="v1"))  # type: ignore
    assert events == [
        {
            "data": {"input": {}},
            "event": "on_tool_start",
            "metadata": {},
            "name": "parameterless",
            "run_id": "",
            "tags": [],
        },
        {
            "data": {"chunk": "hello"},
            "event": "on_tool_stream",
            "metadata": {},
            "name": "parameterless",
            "run_id": "",
            "tags": [],
        },
        {
            "data": {"output": "hello"},
            "event": "on_tool_end",
            "metadata": {},
            "name": "parameterless",
            "run_id": "",
            "tags": [],
        },
    ]

    events = await _collect_events(with_callbacks.astream_events({}, version="v1"))  # type: ignore
    assert events == [
        {
            "data": {"input": {}},
            "event": "on_tool_start",
            "metadata": {},
            "name": "with_callbacks",
            "run_id": "",
            "tags": [],
        },
        {
            "data": {"chunk": "world"},
            "event": "on_tool_stream",
            "metadata": {},
            "name": "with_callbacks",
            "run_id": "",
            "tags": [],
        },
        {
            "data": {"output": "world"},
            "event": "on_tool_end",
            "metadata": {},
            "name": "with_callbacks",
            "run_id": "",
            "tags": [],
        },
    ]
    events = await _collect_events(
        with_parameters.astream_events({"x": 1, "y": "2"}, version="v1")  # type: ignore
    )
    assert events == [
        {
            "data": {"input": {"x": 1, "y": "2"}},
            "event": "on_tool_start",
            "metadata": {},
            "name": "with_parameters",
            "run_id": "",
            "tags": [],
        },
        {
            "data": {"chunk": {"x": 1, "y": "2"}},
            "event": "on_tool_stream",
            "metadata": {},
            "name": "with_parameters",
            "run_id": "",
            "tags": [],
        },
        {
            "data": {"output": {"x": 1, "y": "2"}},
            "event": "on_tool_end",
            "metadata": {},
            "name": "with_parameters",
            "run_id": "",
            "tags": [],
        },
    ]

    events = await _collect_events(
        with_parameters_and_callbacks.astream_events({"x": 1, "y": "2"}, version="v1")  # type: ignore
    )
    assert events == [
        {
            "data": {"input": {"x": 1, "y": "2"}},
            "event": "on_tool_start",
            "metadata": {},
            "name": "with_parameters_and_callbacks",
            "run_id": "",
            "tags": [],
        },
        {
            "data": {"chunk": {"x": 1, "y": "2"}},
            "event": "on_tool_stream",
            "metadata": {},
            "name": "with_parameters_and_callbacks",
            "run_id": "",
            "tags": [],
        },
        {
            "data": {"output": {"x": 1, "y": "2"}},
            "event": "on_tool_end",
            "metadata": {},
            "name": "with_parameters_and_callbacks",
            "run_id": "",
            "tags": [],
        },
    ]


class HardCodedRetriever(BaseRetriever):
    documents: List[Document]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
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
    assert events == [
        {
            "data": {
                "input": {"query": "hello"},
            },
            "event": "on_retriever_start",
            "metadata": {},
            "name": "HardCodedRetriever",
            "run_id": "",
            "tags": [],
        },
        {
            "data": {
                "chunk": [
                    Document(page_content="hello world!", metadata={"foo": "bar"}),
                    Document(page_content="goodbye world!", metadata={"food": "spare"}),
                ]
            },
            "event": "on_retriever_stream",
            "metadata": {},
            "name": "HardCodedRetriever",
            "run_id": "",
            "tags": [],
        },
        {
            "data": {
                "output": [
                    Document(page_content="hello world!", metadata={"foo": "bar"}),
                    Document(page_content="goodbye world!", metadata={"food": "spare"}),
                ],
            },
            "event": "on_retriever_end",
            "metadata": {},
            "name": "HardCodedRetriever",
            "run_id": "",
            "tags": [],
        },
    ]


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

    def format_docs(docs: List[Document]) -> str:
        """Format the docs."""
        return ", ".join([doc.page_content for doc in docs])

    chain = retriever | format_docs
    events = await _collect_events(chain.astream_events("hello", version="v1"))
    assert events == [
        {
            "data": {"input": "hello"},
            "event": "on_chain_start",
            "metadata": {},
            "name": "RunnableSequence",
            "run_id": "",
            "tags": [],
        },
        {
            "data": {"input": {"query": "hello"}},
            "event": "on_retriever_start",
            "metadata": {},
            "name": "Retriever",
            "run_id": "",
            "tags": ["seq:step:1"],
        },
        {
            "data": {
                "input": {"query": "hello"},
                "output": {
                    "documents": [
                        Document(page_content="hello world!", metadata={"foo": "bar"}),
                        Document(
                            page_content="goodbye world!", metadata={"food": "spare"}
                        ),
                    ]
                },
            },
            "event": "on_retriever_end",
            "metadata": {},
            "name": "Retriever",
            "run_id": "",
            "tags": ["seq:step:1"],
        },
        {
            "data": {},
            "event": "on_chain_start",
            "metadata": {},
            "name": "format_docs",
            "run_id": "",
            "tags": ["seq:step:2"],
        },
        {
            "data": {"chunk": "hello world!, goodbye world!"},
            "event": "on_chain_stream",
            "metadata": {},
            "name": "format_docs",
            "run_id": "",
            "tags": ["seq:step:2"],
        },
        {
            "data": {"chunk": "hello world!, goodbye world!"},
            "event": "on_chain_stream",
            "metadata": {},
            "name": "RunnableSequence",
            "run_id": "",
            "tags": [],
        },
        {
            "data": {
                "input": [
                    Document(page_content="hello world!", metadata={"foo": "bar"}),
                    Document(page_content="goodbye world!", metadata={"food": "spare"}),
                ],
                "output": "hello world!, goodbye world!",
            },
            "event": "on_chain_end",
            "metadata": {},
            "name": "format_docs",
            "run_id": "",
            "tags": ["seq:step:2"],
        },
        {
            "data": {"output": "hello world!, goodbye world!"},
            "event": "on_chain_end",
            "metadata": {},
            "name": "RunnableSequence",
            "run_id": "",
            "tags": [],
        },
    ]


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
    chain = concat | reverse  # type: ignore

    events = await _collect_events(
        chain.astream_events({"a": "hello", "b": "world"}, version="v1")
    )
    assert events == [
        {
            "data": {"input": {"a": "hello", "b": "world"}},
            "event": "on_chain_start",
            "metadata": {},
            "name": "RunnableSequence",
            "run_id": "",
            "tags": [],
        },
        {
            "data": {"input": {"a": "hello", "b": "world"}},
            "event": "on_tool_start",
            "metadata": {},
            "name": "concat",
            "run_id": "",
            "tags": ["seq:step:1"],
        },
        {
            "data": {"input": {"a": "hello", "b": "world"}, "output": "helloworld"},
            "event": "on_tool_end",
            "metadata": {},
            "name": "concat",
            "run_id": "",
            "tags": ["seq:step:1"],
        },
        {
            "data": {},
            "event": "on_chain_start",
            "metadata": {},
            "name": "reverse",
            "run_id": "",
            "tags": ["seq:step:2"],
        },
        {
            "data": {"chunk": "dlrowolleh"},
            "event": "on_chain_stream",
            "metadata": {},
            "name": "reverse",
            "run_id": "",
            "tags": ["seq:step:2"],
        },
        {
            "data": {"chunk": "dlrowolleh"},
            "event": "on_chain_stream",
            "metadata": {},
            "name": "RunnableSequence",
            "run_id": "",
            "tags": [],
        },
        {
            "data": {"input": "helloworld", "output": "dlrowolleh"},
            "event": "on_chain_end",
            "metadata": {},
            "name": "reverse",
            "run_id": "",
            "tags": ["seq:step:2"],
        },
        {
            "data": {"output": "dlrowolleh"},
            "event": "on_chain_end",
            "metadata": {},
            "name": "RunnableSequence",
            "run_id": "",
            "tags": [],
        },
    ]


async def test_event_stream_with_retry() -> None:
    """Test the event stream with a tool."""

    def success(inputs: str) -> str:
        return "success"

    def fail(inputs: str) -> None:
        """Simple func."""
        raise Exception("fail")

    chain = RunnableLambda(success) | RunnableLambda(fail).with_retry(
        stop_after_attempt=1,
    )
    iterable = chain.astream_events("q", version="v1")

    events = []

    for _ in range(10):
        try:
            next_chunk = await iterable.__anext__()
            events.append(next_chunk)
        except Exception:
            break

    events = _with_nulled_run_id(events)
    for event in events:
        event["tags"] = sorted(event["tags"])

    assert events == [
        {
            "data": {"input": "q"},
            "event": "on_chain_start",
            "metadata": {},
            "name": "RunnableSequence",
            "run_id": "",
            "tags": [],
        },
        {
            "data": {},
            "event": "on_chain_start",
            "metadata": {},
            "name": "success",
            "run_id": "",
            "tags": ["seq:step:1"],
        },
        {
            "data": {"chunk": "success"},
            "event": "on_chain_stream",
            "metadata": {},
            "name": "success",
            "run_id": "",
            "tags": ["seq:step:1"],
        },
        {
            "data": {},
            "event": "on_chain_start",
            "metadata": {},
            "name": "fail",
            "run_id": "",
            "tags": ["seq:step:2"],
        },
        {
            "data": {"input": "q", "output": "success"},
            "event": "on_chain_end",
            "metadata": {},
            "name": "success",
            "run_id": "",
            "tags": ["seq:step:1"],
        },
        {
            "data": {"input": "success", "output": None},
            "event": "on_chain_end",
            "metadata": {},
            "name": "fail",
            "run_id": "",
            "tags": ["seq:step:2"],
        },
    ]


async def test_with_llm() -> None:
    """Test with regular llm."""
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are Cat Agent 007"), ("human", "{question}")]
    ).with_config({"run_name": "my_template", "tags": ["my_template"]})
    llm = FakeStreamingListLLM(responses=["abc"])

    chain = prompt | llm
    events = await _collect_events(
        chain.astream_events({"question": "hello"}, version="v1")
    )
    assert events == [
        {
            "data": {"input": {"question": "hello"}},
            "event": "on_chain_start",
            "metadata": {},
            "name": "RunnableSequence",
            "run_id": "",
            "tags": [],
        },
        {
            "data": {"input": {"question": "hello"}},
            "event": "on_prompt_start",
            "metadata": {},
            "name": "my_template",
            "run_id": "",
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
            "tags": ["my_template", "seq:step:1"],
        },
        {
            "data": {
                "input": {"prompts": ["System: You are Cat Agent 007\n" "Human: hello"]}
            },
            "event": "on_llm_start",
            "metadata": {},
            "name": "FakeStreamingListLLM",
            "run_id": "",
            "tags": ["seq:step:2"],
        },
        {
            "data": {
                "input": {
                    "prompts": ["System: You are Cat Agent 007\n" "Human: hello"]
                },
                "output": {
                    "generations": [
                        [{"generation_info": None, "text": "abc", "type": "Generation"}]
                    ],
                    "llm_output": None,
                    "run": None,
                },
            },
            "event": "on_llm_end",
            "metadata": {},
            "name": "FakeStreamingListLLM",
            "run_id": "",
            "tags": ["seq:step:2"],
        },
        {
            "data": {"chunk": "a"},
            "event": "on_chain_stream",
            "metadata": {},
            "name": "RunnableSequence",
            "run_id": "",
            "tags": [],
        },
        {
            "data": {"chunk": "b"},
            "event": "on_chain_stream",
            "metadata": {},
            "name": "RunnableSequence",
            "run_id": "",
            "tags": [],
        },
        {
            "data": {"chunk": "c"},
            "event": "on_chain_stream",
            "metadata": {},
            "name": "RunnableSequence",
            "run_id": "",
            "tags": [],
        },
        {
            "data": {"output": "abc"},
            "event": "on_chain_end",
            "metadata": {},
            "name": "RunnableSequence",
            "run_id": "",
            "tags": [],
        },
    ]


async def test_runnable_each() -> None:
    """Test runnable each astream_events."""

    async def add_one(x: int) -> int:
        return x + 1

    add_one_map = RunnableLambda(add_one).map()  # type: ignore
    assert await add_one_map.ainvoke([1, 2, 3]) == [2, 3, 4]

    with pytest.raises(NotImplementedError):
        async for _ in add_one_map.astream_events([1, 2, 3], version="v1"):
            pass


async def test_events_astream_config() -> None:
    """Test that astream events support accepting config"""
    infinite_cycle = cycle([AIMessage(content="hello world!")])
    good_world_on_repeat = cycle([AIMessage(content="Goodbye world")])
    model = GenericFakeChatModel(messages=infinite_cycle).configurable_fields(
        messages=ConfigurableField(
            id="messages",
            name="Messages",
            description="Messages return by the LLM",
        )
    )

    model_02 = model.with_config({"configurable": {"messages": good_world_on_repeat}})
    assert model_02.invoke("hello") == AIMessage(content="Goodbye world")

    events = await _collect_events(model_02.astream_events("hello", version="v1"))
    assert events == [
        {
            "data": {"input": "hello"},
            "event": "on_chat_model_start",
            "metadata": {},
            "name": "RunnableConfigurableFields",
            "run_id": "",
            "tags": [],
        },
        {
            "data": {"chunk": AIMessageChunk(content="Goodbye")},
            "event": "on_chat_model_stream",
            "metadata": {},
            "name": "RunnableConfigurableFields",
            "run_id": "",
            "tags": [],
        },
        {
            "data": {"chunk": AIMessageChunk(content=" ")},
            "event": "on_chat_model_stream",
            "metadata": {},
            "name": "RunnableConfigurableFields",
            "run_id": "",
            "tags": [],
        },
        {
            "data": {"chunk": AIMessageChunk(content="world")},
            "event": "on_chat_model_stream",
            "metadata": {},
            "name": "RunnableConfigurableFields",
            "run_id": "",
            "tags": [],
        },
        {
            "data": {"output": AIMessageChunk(content="Goodbye world")},
            "event": "on_chat_model_end",
            "metadata": {},
            "name": "RunnableConfigurableFields",
            "run_id": "",
            "tags": [],
        },
    ]


async def test_runnable_with_message_history() -> None:
    class InMemoryHistory(BaseChatMessageHistory, BaseModel):
        """In memory implementation of chat message history."""

        # Attention: for the tests use an Any type to work-around a pydantic issue
        # where it re-instantiates a list, so mutating the list doesn't end up mutating
        # the content in the store!

        # Using Any type here rather than List[BaseMessage] due to pydantic issue!
        messages: Any

        def add_message(self, message: BaseMessage) -> None:
            """Add a self-created message to the store."""
            self.messages.append(message)

        def clear(self) -> None:
            self.messages = []

    # Here we use a global variable to store the chat message history.
    # This will make it easier to inspect it to see the underlying results.
    store: Dict = {}

    def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
        """Get a chat message history"""
        if session_id not in store:
            store[session_id] = []
        return InMemoryHistory(messages=store[session_id])

    infinite_cycle = cycle([AIMessage(content="hello"), AIMessage(content="world")])

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a cat"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )
    model = GenericFakeChatModel(messages=infinite_cycle)

    chain: Runnable = prompt | model
    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history=get_by_session_id,
        input_messages_key="question",
        history_messages_key="history",
    )
    with_message_history.with_config(
        {"configurable": {"session_id": "session-123"}}
    ).invoke({"question": "hello"})

    assert store == {
        "session-123": [HumanMessage(content="hello"), AIMessage(content="hello")]
    }

    with_message_history.with_config(
        {"configurable": {"session_id": "session-123"}}
    ).invoke({"question": "meow"})
    assert store == {
        "session-123": [
            HumanMessage(content="hello"),
            AIMessage(content="hello"),
            HumanMessage(content="meow"),
            AIMessage(content="world"),
        ]
    }
