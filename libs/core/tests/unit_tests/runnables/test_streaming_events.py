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
from langchain_core.runnables.utils import Event, Input, Output, as_event_stream
from langchain_core.tools import tool
from langchain_core.tracers import RunLog, RunLogPatch
from tests.unit_tests.fake.chat_model import GenericFakeChatModel


def _with_nulled_run_id(events: Sequence[Event]) -> List[Event]:
    """Removes the run ids from events."""
    return [{**event, "run_id": None} for event in events]


async def _as_async_iterator(iterable: List) -> AsyncIterator:
    """Converts an iterable into an async iterator."""
    for item in iterable:
        yield item


async def _get_events(run_log_patches: AsyncIterator[RunLogPatch]) -> List[Event]:
    """A helper function to facilitate testing.

    The goal is to consume the astream log, and then refeed it into the
    as_event_stream function. This helps makes sure that exceptions in the
    as_event_stream function are surfaced in an obvious way.

    In addition, the run ids are nulled out so that the test is not dependent
    on the run id (which is randomly generated).
    """
    run_log_patches = [patch async for patch in run_log_patches]
    events = [
        event async for event in as_event_stream(_as_async_iterator(run_log_patches))
    ]

    for event in events:
        event["tags"] = sorted(event["tags"])
    return _with_nulled_run_id(events)


async def test_event_stream_with_simple_chain() -> None:
    """Test as event stream."""
    template = ChatPromptTemplate.from_messages(
        [("system", "You are Cat Agent 007"), ("human", "{question}")]
    ).with_config({"run_name": "my_template", "tags": ["my_template"]})

    infinite_cycle = cycle(
        [AIMessage(content="hello world!"), AIMessage(content="goodbye world!")]
    )
    # When streaming GenericFakeChatModel breaks AIMessage into chunks based on spaces
    model = GenericFakeChatModel(messages=infinite_cycle).with_config(
        {
            "metadata": {"a": "b"},
            "tags": ["my_model"],
            "run_name": "my_model",
        }
    )

    chain = (template | model).with_config(
        {
            "metadata": {"foo": "bar"},
            "tags": ["my_chain"],
            "run_name": "my_chain",
        }
    )

    events = await _get_events(chain.astream_log({"question": "hello"}))
    assert events == [
        {
            "data": {"inputs": {"question": "hello"}},
            "event": "on_prompt_start",
            "metadata": {"foo": "bar"},
            "name": "my_template",
            "run_id": None,
            "tags": ["my_chain", "my_template", "seq:step:1"],
        },
        {
            "data": {
                "output": ChatPromptValue(
                    messages=[
                        SystemMessage(content="You are Cat Agent 007"),
                        HumanMessage(content="hello"),
                    ]
                )
            },
            "event": "on_prompt_end",
            "metadata": {"foo": "bar"},
            "name": "my_template",
            "run_id": None,
            "tags": ["my_chain", "my_template", "seq:step:1"],
        },
        {
            "data": {
                "inputs": {
                    "prompts": ["System: You are Cat Agent 007\n" "Human: hello"]
                }
            },
            "event": "on_llm_start",
            "metadata": {"a": "b", "foo": "bar"},
            "name": "my_model",
            "run_id": None,
            "tags": ["my_chain", "my_model", "seq:step:2"],
        },
        {
            "data": [AIMessageChunk(content="hello")],
            "event": "on_llm_stream",
            "metadata": {"a": "b", "foo": "bar"},
            "name": "my_model",
            "run_id": None,
            "tags": ["my_chain", "my_model", "seq:step:2"],
        },
        {
            "data": [AIMessageChunk(content=" ")],
            "event": "on_llm_stream",
            "metadata": {"a": "b", "foo": "bar"},
            "name": "my_model",
            "run_id": None,
            "tags": ["my_chain", "my_model", "seq:step:2"],
        },
        {
            "data": [AIMessageChunk(content="world!")],
            "event": "on_llm_stream",
            "metadata": {"a": "b", "foo": "bar"},
            "name": "my_model",
            "run_id": None,
            "tags": ["my_chain", "my_model", "seq:step:2"],
        },
        {
            "data": {
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
                }
            },
            "event": "on_llm_end",
            "metadata": {"a": "b", "foo": "bar"},
            "name": "my_model",
            "run_id": None,
            "tags": ["my_chain", "my_model", "seq:step:2"],
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
    iterable = chain.astream_log({})

    chunks = []

    for _ in range(10):
        try:
            next_chunk = await iterable.__anext__()
            chunks.append(next_chunk)
        except Exception:
            break

    assert await _get_events(_as_async_iterator(chunks)) == [
        {
            "data": {
                "inputs": {
                    "input": ""
                }  # TODO: Why is this an empty string? Should be empty dict
            },
            "event": "on_chain_start",
            "metadata": {},
            "name": "success",
            "run_id": None,
            "tags": ["seq:step:1"],
        },
        {
            "data": ["success"],
            "event": "on_chain_stream",
            "metadata": {},
            "name": "success",
            "run_id": None,
            "tags": ["seq:step:1"],
        },
        {
            "data": {"inputs": {"input": ""}},
            "event": "on_chain_start",
            "metadata": {},
            "name": "fail",
            "run_id": None,
            "tags": ["seq:step:2"],
        },
        {
            "data": {"output": {"output": "success"}},
            "event": "on_chain_end",
            "metadata": {},
            "name": "success",
            "run_id": None,
            "tags": ["seq:step:1"],
        },
        {
            "data": {},
            "event": "on_chain_end",
            "metadata": {},
            "name": "fail",
            "run_id": None,
            "tags": ["seq:step:2"],
        },
    ]


async def _as_run_log_state(run_log_patches: Sequence[RunLogPatch]) -> RunLog:
    """Converts a sequence of run log patches into a run log state."""
    state = RunLog(state=None)
    async for run_log_patch in run_log_patches:
        state = state + run_log_patch
    return state


async def test_event_stream_with_lambdas() -> None:
    """Test event stream with lamdbas."""
    as_lambdas = RunnableLambda(lambda x: {"answer": "goodbye"})

    events = await _get_events(as_lambdas.astream_log({"question": "hello"}))
    assert len(events) == 3
    assert events == [
        {
            "data": {"input": ""},
            "event": "on_root_start",
            "metadata": {},
            "name": "placeholder",
            "run_id": None,
            "tags": [],
        },
        {
            "data": [{"answer": "goodbye"}],
            "event": "on__root__stream",
            "metadata": {},
            "name": "placeholder",
            "run_id": None,
            "tags": [],
        },
        {
            "data": {"answer": "goodbye"},
            "event": "on__root__end",
            "metadata": {},
            "name": "placeholder",
            "run_id": None,
            "tags": [],
        },
    ]

    def add_one(x: int) -> int:
        """Add one to x."""
        return x + 1

    events = await _get_events(RunnableLambda(add_one).astream_log(1))
    assert len(events) == 3
    assert events == [
        {
            "data": {"input": ""},
            "event": "on_root_start",
            "metadata": {},
            "name": "placeholder",
            "run_id": None,
            "tags": [],
        },
        {
            "data": [2],
            "event": "on__root__stream",
            "metadata": {},
            "name": "placeholder",
            "run_id": None,
            "tags": [],
        },
        {
            "data": 2,
            "event": "on__root__end",
            "metadata": {},
            "name": "placeholder",
            "run_id": None,
            "tags": [],
        },
    ]


async def test_event_stream_with_tool() -> None:
    """Test the event stream with a tool."""
    assert "Fails for some reason" == "Some json patch thingy?"

    @tool
    def say_what() -> str:
        """A tool that does nothing."""
        return "what"

    class CustomRunnable(Runnable):
        """A custom runnable that uses the tool."""

        def invoke(
            self, input: Input, config: Optional[RunnableConfig] = None
        ) -> Output:
            raise NotImplementedError()

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

    events = await _get_events(custom_runnable.astream_log({}))
    assert events == []


class HardCodedRetriever(BaseRetriever):
    documents: List[Document]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self.documents


async def test_event_stream_with_retriever() -> None:
    """Test the event stream with a retriever."""
    assert "Fails due to missing root run information" == "TODO: Fix this test"
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
    events = await _get_events(retriever.astream_log({"query": "hello"}))
    assert events == []


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

    events = await _get_events(chain.astream_log("hello"))
    assert events == [
        {
            "data": {"inputs": {"query": "hello"}},
            "event": "on_retriever_start",
            "metadata": {},
            "name": "Retriever",
            "run_id": None,
            "tags": ["seq:step:1"],
        },
        {
            "data": {
                "output": {
                    # TODO: This output is likely overly nested
                    "documents": [
                        Document(page_content="hello world!", metadata={"foo": "bar"}),
                        Document(
                            page_content="goodbye world!", metadata={"food": "spare"}
                        ),
                    ]
                }
            },
            "event": "on_retriever_end",
            "metadata": {},
            "name": "Retriever",
            "run_id": None,
            "tags": ["seq:step:1"],
        },
        {
            "data": {"inputs": {"input": ""}},
            "event": "on_chain_start",
            "metadata": {},
            "name": "format_docs",
            "run_id": None,
            "tags": ["seq:step:2"],
        },
        {
            "data": ["hello world!, goodbye world!"],  # TODO: How??
            "event": "on_chain_stream",
            "metadata": {},
            "name": "format_docs",
            "run_id": None,
            "tags": ["seq:step:2"],
        },
        {
            "data": {"output": {"output": "hello world!, goodbye world!"}},
            "event": "on_chain_end",
            "metadata": {},
            "name": "format_docs",
            "run_id": None,
            "tags": ["seq:step:2"],
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

    assert chain.invoke({"a": "hello", "b": "world"}) == "dlrowolleh"

    events = await _get_events(chain.astream_log({"a": "hello", "b": "world"}))
    assert events == [
        {
            "data": {"inputs": {"input": "{'a': 'hello', 'b': 'world'}"}},  # WHAT??
            "event": "on_tool_start",
            "metadata": {},
            "name": "concat",
            "run_id": None,
            "tags": ["seq:step:1"],
        },
        {
            "data": {"output": {"output": "helloworld"}},
            "event": "on_tool_end",
            "metadata": {},
            "name": "concat",
            "run_id": None,
            "tags": ["seq:step:1"],
        },
        {
            "data": {"inputs": {"input": ""}},
            "event": "on_chain_start",
            "metadata": {},
            "name": "reverse",
            "run_id": None,
            "tags": ["seq:step:2"],
        },
        {
            "data": ["dlrowolleh"],
            "event": "on_chain_stream",
            "metadata": {},
            "name": "reverse",
            "run_id": None,
            "tags": ["seq:step:2"],
        },
        {
            "data": {"output": {"output": "dlrowolleh"}},
            "event": "on_chain_end",
            "metadata": {},
            "name": "reverse",
            "run_id": None,
            "tags": ["seq:step:2"],
        },
    ]
