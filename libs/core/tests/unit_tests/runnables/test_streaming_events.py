from itertools import cycle
from typing import Any, AsyncIterator, List, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.runnables.utils import Input, Output, StreamEvent
from langchain_core.tools import tool
from langchain_core.tracers import RunLog, RunLogPatch
from tests.unit_tests.fake.chat_model import GenericFakeChatModel


def _with_nulled_run_id(events: Sequence[StreamEvent]) -> List[StreamEvent]:
    """Removes the run ids from events."""
    return [{**event, "run_id": None} for event in events]


async def _as_async_iterator(iterable: List) -> AsyncIterator:
    """Converts an iterable into an async iterator."""
    for item in iterable:
        yield item


async def _collect_events(events: AsyncIterator[StreamEvent]) -> List[StreamEvent]:
    """Collect the events and remove the run ids."""
    events = [event async for event in events]
    events = _with_nulled_run_id(events)
    for event in events:
        event["tags"] = sorted(event["tags"])
    return events


async def _as_run_log_state(run_log_patches: Sequence[RunLogPatch]) -> dict:
    """Converts a sequence of run log patches into a run log state."""
    state = RunLog(state=None)
    async for run_log_patch in run_log_patches:
        state = state + run_log_patch
    return state


async def test_event_stream_with_single_lambda() -> None:
    """Test the event stream with a tool."""

    def reverse(s: str) -> str:
        """Reverse a string."""
        return s[::-1]

    chain = RunnableLambda(func=reverse)

    events = await _collect_events(chain.astream_events("hello"))
    assert events == [
        {
            "data": {"input": "hello"},
            "event": "on_chain_start",
            "metadata": {},
            "name": "reverse",
            "run_id": None,
            "tags": [],
        },
        {
            "data": {"chunk": "olleh"},
            "event": "on_chain_stream",
            "metadata": {},
            "name": "reverse",
            "run_id": None,
            "tags": [],
        },
        {
            "data": {"output": "olleh"},
            "event": "on_chain_end",
            "metadata": {},
            "name": "reverse",
            "run_id": None,
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
    events = await _collect_events(chain.astream_events("hello"))
    assert events == [
        {
            "data": {"input": "hello"},
            "event": "on_chain_start",
            "metadata": {},
            "name": "RunnableSequence",
            "run_id": None,
            "tags": [],
        },
        {
            "data": {},
            "event": "on_chain_start",
            "metadata": {},
            "name": "1",
            "run_id": None,
            "tags": ["seq:step:1"],
        },
        {
            "data": {"chunk": "olleh"},
            "event": "on_chain_stream",
            "metadata": {},
            "name": "1",
            "run_id": None,
            "tags": ["seq:step:1"],
        },
        {
            "data": {},
            "event": "on_chain_start",
            "metadata": {},
            "name": "2",
            "run_id": None,
            "tags": ["seq:step:2"],
        },
        {
            "data": {"input": "hello", "output": "olleh"},
            "event": "on_chain_end",
            "metadata": {},
            "name": "1",
            "run_id": None,
            "tags": ["seq:step:1"],
        },
        {
            "data": {"chunk": "hello"},
            "event": "on_chain_stream",
            "metadata": {},
            "name": "2",
            "run_id": None,
            "tags": ["seq:step:2"],
        },
        {
            "data": {},
            "event": "on_chain_start",
            "metadata": {},
            "name": "3",
            "run_id": None,
            "tags": ["seq:step:3"],
        },
        {
            "data": {"input": "olleh", "output": "hello"},
            "event": "on_chain_end",
            "metadata": {},
            "name": "2",
            "run_id": None,
            "tags": ["seq:step:2"],
        },
        {
            "data": {"chunk": "olleh"},
            "event": "on_chain_stream",
            "metadata": {},
            "name": "3",
            "run_id": None,
            "tags": ["seq:step:3"],
        },
        {
            "data": {"chunk": "olleh"},
            "event": "on_chain_stream",
            "metadata": {},
            "name": "RunnableSequence",
            "run_id": None,
            "tags": [],
        },
        {
            "data": {"input": "hello", "output": "olleh"},
            "event": "on_chain_end",
            "metadata": {},
            "name": "3",
            "run_id": None,
            "tags": ["seq:step:3"],
        },
        {
            "data": {"output": "olleh"},
            "event": "on_chain_end",
            "metadata": {},
            "name": "RunnableSequence",
            "run_id": None,
            "tags": [],
        },
    ]


async def test_event_stream_with_lambdas_from_lambda() -> None:
    as_lambdas = RunnableLambda(lambda x: {"answer": "goodbye"}).with_config(
        {"run_name": "my_lambda"}
    )
    events = await _collect_events(as_lambdas.astream_events({"question": "hello"}))
    assert events == [
        {
            "data": {"input": {"question": "hello"}},
            "event": "on_chain_start",
            "metadata": {},
            "name": "my_lambda",
            "run_id": None,
            "tags": [],
        },
        {
            "data": {"chunk": {"answer": "goodbye"}},
            "event": "on_chain_stream",
            "metadata": {},
            "name": "my_lambda",
            "run_id": None,
            "tags": [],
        },
        {
            "data": {"output": {"answer": "goodbye"}},
            "event": "on_chain_end",
            "metadata": {},
            "name": "my_lambda",
            "run_id": None,
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

    events = await _collect_events(chain.astream_events({"question": "hello"}))
    assert events == [
        {
            "data": {"input": {"question": "hello"}},
            "event": "on_chain_start",
            "metadata": {},
            "name": "my_chain",
            "run_id": None,
            "tags": [],
        },
        {
            "data": {"input": {"question": "hello"}},
            "event": "on_prompt_start",
            "metadata": {"foo": "bar"},
            "name": "my_template",
            "run_id": None,
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
            "run_id": None,
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
            "event": "on_llm_start",
            "metadata": {"a": "b", "foo": "bar"},
            "name": "my_model",
            "run_id": None,
            "tags": ["my_chain", "my_model", "seq:step:2"],
        },
        {
            "data": {"chunk": AIMessageChunk(content="hello")},
            "event": "on_chain_stream",
            "metadata": {},
            "name": "my_chain",
            "run_id": None,
            "tags": [],
        },
        {
            "data": {"chunk": AIMessageChunk(content="hello")},
            "event": "on_llm_stream",
            "metadata": {"a": "b", "foo": "bar"},
            "name": "my_model",
            "run_id": None,
            "tags": ["my_chain", "my_model", "seq:step:2"],
        },
        {
            "data": {"chunk": AIMessageChunk(content=" ")},
            "event": "on_chain_stream",
            "metadata": {},
            "name": "my_chain",
            "run_id": None,
            "tags": [],
        },
        {
            "data": {"chunk": AIMessageChunk(content=" ")},
            "event": "on_llm_stream",
            "metadata": {"a": "b", "foo": "bar"},
            "name": "my_model",
            "run_id": None,
            "tags": ["my_chain", "my_model", "seq:step:2"],
        },
        {
            "data": {"chunk": AIMessageChunk(content="world!")},
            "event": "on_chain_stream",
            "metadata": {},
            "name": "my_chain",
            "run_id": None,
            "tags": [],
        },
        {
            "data": {"chunk": AIMessageChunk(content="world!")},
            "event": "on_llm_stream",
            "metadata": {"a": "b", "foo": "bar"},
            "name": "my_model",
            "run_id": None,
            "tags": ["my_chain", "my_model", "seq:step:2"],
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
            "event": "on_llm_end",
            "metadata": {"a": "b", "foo": "bar"},
            "name": "my_model",
            "run_id": None,
            "tags": ["my_chain", "my_model", "seq:step:2"],
        },
        {
            "data": {
                "output": AIMessageChunk(content="hello world!"),
            },
            "event": "on_chain_end",
            "metadata": {},
            "name": "my_chain",
            "run_id": None,
            "tags": [],
        },
    ]


async def test_event_stream_with_tool() -> None:
    """Test the event stream with a tool."""

    @tool
    def say_what() -> str:
        """A tool that does nothing."""
        return "what"

    class CustomRunnable(Runnable):
        """A custom runnable that uses the tool."""

        def invoke(
            self, input: Input, config: Optional[RunnableConfig] = None
        ) -> Output:
            return "hello"

        async def astream(
            self,
            input: Input,
            config: Optional[RunnableConfig] = None,
            **kwargs: Optional[Any],
        ) -> AsyncIterator[Output]:
            """A custom async stream."""
            result = say_what.run({"foo": "bar"})
            for char in result:
                yield char

    custom_runnable = CustomRunnable().with_config(
        {
            "metadata": {"foo": "bar"},
            "tags": ["my_runnable"],
            "run_name": "my_runnable",
        }
    )

    state = RunLog(state=None)

    async for run_log_patch in custom_runnable.astream_log({}):
        state = state + run_log_patch


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
    events = await _collect_events(retriever.astream_events({"query": "hello"}))
    assert events == [
        {
            "data": {
                "input": {"query": "hello"},
            },
            "event": "on_retriever_start",
            "metadata": {},
            "name": "Retriever",
            "run_id": None,
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
            "name": "Retriever",
            "run_id": None,
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
            "name": "Retriever",
            "run_id": None,
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
    events = await _collect_events(chain.astream_events("hello"))
    assert events == [
        {
            "data": {"input": "hello"},
            "event": "on_chain_start",
            "metadata": {},
            "name": "RunnableSequence",
            "run_id": None,
            "tags": [],
        },
        {
            "data": {"input": {"query": "hello"}},
            "event": "on_retriever_start",
            "metadata": {},
            "name": "Retriever",
            "run_id": None,
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
            "run_id": None,
            "tags": ["seq:step:1"],
        },
        {
            "data": {},
            "event": "on_chain_start",
            "metadata": {},
            "name": "format_docs",
            "run_id": None,
            "tags": ["seq:step:2"],
        },
        {
            "data": {"chunk": "hello world!, goodbye world!"},
            "event": "on_chain_stream",
            "metadata": {},
            "name": "format_docs",
            "run_id": None,
            "tags": ["seq:step:2"],
        },
        {
            "data": {"chunk": "hello world!, goodbye world!"},
            "event": "on_chain_stream",
            "metadata": {},
            "name": "RunnableSequence",
            "run_id": None,
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
            "run_id": None,
            "tags": ["seq:step:2"],
        },
        {
            "data": {"output": "hello world!, goodbye world!"},
            "event": "on_chain_end",
            "metadata": {},
            "name": "RunnableSequence",
            "run_id": None,
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

    chain = concat | reverse

    events = await _collect_events(chain.astream_events({"a": "hello", "b": "world"}))
    assert events == [
        {
            "data": {"input": {"a": "hello", "b": "world"}},
            "event": "on_chain_start",
            "metadata": {},
            "name": "RunnableSequence",
            "run_id": None,
            "tags": [],
        },
        {
            "data": {"input": {"a": "hello", "b": "world"}},
            "event": "on_tool_start",
            "metadata": {},
            "name": "concat",
            "run_id": None,
            "tags": ["seq:step:1"],
        },
        {
            "data": {"input": {"a": "hello", "b": "world"}, "output": "helloworld"},
            "event": "on_tool_end",
            "metadata": {},
            "name": "concat",
            "run_id": None,
            "tags": ["seq:step:1"],
        },
        {
            "data": {},
            "event": "on_chain_start",
            "metadata": {},
            "name": "reverse",
            "run_id": None,
            "tags": ["seq:step:2"],
        },
        {
            "data": {"chunk": "dlrowolleh"},
            "event": "on_chain_stream",
            "metadata": {},
            "name": "reverse",
            "run_id": None,
            "tags": ["seq:step:2"],
        },
        {
            "data": {"chunk": "dlrowolleh"},
            "event": "on_chain_stream",
            "metadata": {},
            "name": "RunnableSequence",
            "run_id": None,
            "tags": [],
        },
        {
            "data": {"input": "helloworld", "output": "dlrowolleh"},
            "event": "on_chain_end",
            "metadata": {},
            "name": "reverse",
            "run_id": None,
            "tags": ["seq:step:2"],
        },
        {
            "data": {"output": "dlrowolleh"},
            "event": "on_chain_end",
            "metadata": {},
            "name": "RunnableSequence",
            "run_id": None,
            "tags": [],
        },
    ]


async def test_event_stream_with_retry() -> None:
    """Test the event stream with a tool."""

    def success(inputs) -> str:
        return "success"

    def fail(inputs) -> None:
        """Simple func."""
        raise Exception("fail")

    chain = RunnableLambda(success) | RunnableLambda(fail).with_retry(
        stop_after_attempt=1,
    )
    iterable = chain.astream_events("q")

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
            "run_id": None,
            "tags": [],
        },
        {
            "data": {},
            "event": "on_chain_start",
            "metadata": {},
            "name": "success",
            "run_id": None,
            "tags": ["seq:step:1"],
        },
        {
            "data": {"chunk": "success"},
            "event": "on_chain_stream",
            "metadata": {},
            "name": "success",
            "run_id": None,
            "tags": ["seq:step:1"],
        },
        {
            "data": {},
            "event": "on_chain_start",
            "metadata": {},
            "name": "fail",
            "run_id": None,
            "tags": ["seq:step:2"],
        },
        {
            "data": {"input": "q", "output": "success"},
            "event": "on_chain_end",
            "metadata": {},
            "name": "success",
            "run_id": None,
            "tags": ["seq:step:1"],
        },
        {
            "data": {"input": "success", "output": None},
            "event": "on_chain_end",
            "metadata": {},
            "name": "fail",
            "run_id": None,
            "tags": ["seq:step:2"],
        },
    ]
