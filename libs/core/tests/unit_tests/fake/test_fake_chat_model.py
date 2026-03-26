"""Tests for verifying that testing utility code works as expected."""

import time
from itertools import cycle
from typing import Any, cast
from uuid import UUID

import pytest
from pydantic import BaseModel
from typing_extensions import override

from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.language_models import (
    FakeListChatModel,
    FakeMessagesListChatModel,
    GenericFakeChatModel,
    ParrotFakeChatModel,
)
from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk
from tests.unit_tests.stubs import (
    _any_id_ai_message,
    _any_id_ai_message_chunk,
    _any_id_human_message,
)


def test_generic_fake_chat_model_invoke() -> None:
    # Will alternate between responding with hello and goodbye
    infinite_cycle = cycle([AIMessage(content="hello"), AIMessage(content="goodbye")])
    model = GenericFakeChatModel(messages=infinite_cycle)
    response = model.invoke("meow")
    assert response == _any_id_ai_message(content="hello")
    response = model.invoke("kitty")
    assert response == _any_id_ai_message(content="goodbye")
    response = model.invoke("meow")
    assert response == _any_id_ai_message(content="hello")


async def test_generic_fake_chat_model_ainvoke() -> None:
    # Will alternate between responding with hello and goodbye
    infinite_cycle = cycle([AIMessage(content="hello"), AIMessage(content="goodbye")])
    model = GenericFakeChatModel(messages=infinite_cycle)
    response = await model.ainvoke("meow")
    assert response == _any_id_ai_message(content="hello")
    response = await model.ainvoke("kitty")
    assert response == _any_id_ai_message(content="goodbye")
    response = await model.ainvoke("meow")
    assert response == _any_id_ai_message(content="hello")


async def test_generic_fake_chat_model_stream() -> None:
    """Test streaming."""
    infinite_cycle = cycle(
        [
            AIMessage(content="hello goodbye"),
        ]
    )
    model = GenericFakeChatModel(messages=infinite_cycle)
    chunks = [chunk async for chunk in model.astream("meow")]
    assert chunks == [
        _any_id_ai_message_chunk(content="hello"),
        _any_id_ai_message_chunk(content=" "),
        _any_id_ai_message_chunk(content="goodbye", chunk_position="last"),
    ]
    assert len({chunk.id for chunk in chunks}) == 1

    chunks = list(model.stream("meow"))
    assert chunks == [
        _any_id_ai_message_chunk(content="hello"),
        _any_id_ai_message_chunk(content=" "),
        _any_id_ai_message_chunk(content="goodbye", chunk_position="last"),
    ]
    assert len({chunk.id for chunk in chunks}) == 1

    # Test streaming of additional kwargs.
    # Relying on insertion order of the additional kwargs dict
    message = AIMessage(content="", additional_kwargs={"foo": 42, "bar": 24})
    model = GenericFakeChatModel(messages=cycle([message]))
    chunks = [chunk async for chunk in model.astream("meow")]
    assert chunks == [
        _any_id_ai_message_chunk(content="", additional_kwargs={"foo": 42}),
        _any_id_ai_message_chunk(content="", additional_kwargs={"bar": 24}),
        _any_id_ai_message_chunk(content="", chunk_position="last"),
    ]
    assert len({chunk.id for chunk in chunks}) == 1

    message = AIMessage(
        content="",
        additional_kwargs={
            "function_call": {
                "name": "move_file",
                "arguments": '{\n  "source_path": "foo",\n  "'
                'destination_path": "bar"\n}',
            }
        },
    )
    model = GenericFakeChatModel(messages=cycle([message]))
    chunks = [chunk async for chunk in model.astream("meow")]

    assert chunks == [
        _any_id_ai_message_chunk(
            content="",
            additional_kwargs={"function_call": {"name": "move_file"}},
        ),
        _any_id_ai_message_chunk(
            content="",
            additional_kwargs={
                "function_call": {"arguments": '{\n  "source_path": "foo"'},
            },
        ),
        _any_id_ai_message_chunk(
            content="", additional_kwargs={"function_call": {"arguments": ","}}
        ),
        _any_id_ai_message_chunk(
            content="",
            additional_kwargs={
                "function_call": {"arguments": '\n  "destination_path": "bar"\n}'},
            },
        ),
        _any_id_ai_message_chunk(content="", chunk_position="last"),
    ]
    assert len({chunk.id for chunk in chunks}) == 1

    accumulate_chunks = None
    for chunk in chunks:
        if accumulate_chunks is None:
            accumulate_chunks = chunk
        else:
            accumulate_chunks += chunk

    assert accumulate_chunks == AIMessageChunk(
        content="",
        additional_kwargs={
            "function_call": {
                "name": "move_file",
                "arguments": '{\n  "source_path": "foo",\n  "'
                'destination_path": "bar"\n}',
            }
        },
        id=chunks[0].id,
        chunk_position="last",
    )


async def test_generic_fake_chat_model_astream_log() -> None:
    """Test streaming."""
    infinite_cycle = cycle([AIMessage(content="hello goodbye")])
    model = GenericFakeChatModel(messages=infinite_cycle)
    log_patches = [
        log_patch async for log_patch in model.astream_log("meow", diff=False)
    ]
    final = log_patches[-1]
    assert final.state["streamed_output"] == [
        _any_id_ai_message_chunk(content="hello"),
        _any_id_ai_message_chunk(content=" "),
        _any_id_ai_message_chunk(content="goodbye", chunk_position="last"),
    ]
    assert len({chunk.id for chunk in final.state["streamed_output"]}) == 1


async def test_callback_handlers() -> None:
    """Verify that model is implemented correctly with handlers working."""

    class MyCustomAsyncHandler(AsyncCallbackHandler):
        def __init__(self, store: list[str]) -> None:
            self.store = store

        async def on_chat_model_start(
            self,
            serialized: dict[str, Any],
            messages: list[list[BaseMessage]],
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            tags: list[str] | None = None,
            metadata: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> Any:
            # Do nothing
            # Required to implement since this is an abstract method
            pass

        @override
        async def on_llm_new_token(
            self,
            token: str,
            *,
            chunk: GenerationChunk | ChatGenerationChunk | None = None,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            tags: list[str] | None = None,
            **kwargs: Any,
        ) -> None:
            self.store.append(token)

    infinite_cycle = cycle(
        [
            AIMessage(content="hello goodbye"),
        ]
    )
    model = GenericFakeChatModel(messages=infinite_cycle)
    tokens: list[str] = []
    # New model
    results = [
        chunk
        async for chunk in model.astream(
            "meow", {"callbacks": [MyCustomAsyncHandler(tokens)]}
        )
    ]
    assert results == [
        _any_id_ai_message_chunk(content="hello"),
        _any_id_ai_message_chunk(content=" "),
        _any_id_ai_message_chunk(content="goodbye", chunk_position="last"),
    ]
    assert tokens == ["hello", " ", "goodbye"]
    assert len({chunk.id for chunk in results}) == 1


def test_chat_model_inputs() -> None:
    fake = ParrotFakeChatModel()

    assert cast("HumanMessage", fake.invoke("hello")) == _any_id_human_message(
        content="hello"
    )
    assert fake.invoke([("ai", "blah")]) == _any_id_ai_message(content="blah")
    assert fake.invoke([AIMessage(content="blah")]) == _any_id_ai_message(
        content="blah"
    )


def test_fake_list_chat_model_batch() -> None:
    expected = [
        _any_id_ai_message(content="a"),
        _any_id_ai_message(content="b"),
        _any_id_ai_message(content="c"),
    ]
    for _ in range(20):
        # run this 20 times to test race condition in batch
        fake = FakeListChatModel(responses=["a", "b", "c"])
        resp = fake.batch(["1", "2", "3"])
        assert resp == expected


def test_fake_messages_list_chat_model_sleep_delay() -> None:
    sleep_time = 0.1
    model = FakeMessagesListChatModel(
        responses=[AIMessage(content="A"), AIMessage(content="B")],
        sleep=sleep_time,
    )
    messages = [HumanMessage(content="C")]

    start = time.time()
    model.invoke(messages)
    elapsed = time.time() - start

    assert elapsed >= sleep_time


# --- Tests for bind_tools and with_structured_output ---


class Person(BaseModel):
    """A person."""

    name: str
    age: int


def _make_fake_model(
    cls: type,
) -> (
    FakeMessagesListChatModel
    | FakeListChatModel
    | FakeChatModel
    | GenericFakeChatModel
    | ParrotFakeChatModel
):
    """Construct a fake model instance for the given class."""
    if cls is FakeMessagesListChatModel:
        return FakeMessagesListChatModel(responses=[AIMessage(content="hi")])
    if cls is FakeListChatModel:
        return FakeListChatModel(responses=["hi"])
    if cls is FakeChatModel:
        return FakeChatModel()
    if cls is GenericFakeChatModel:
        return GenericFakeChatModel(messages=iter([AIMessage(content="hi")]))
    if cls is ParrotFakeChatModel:
        return ParrotFakeChatModel()
    msg = f"Unknown fake model class: {cls}"
    raise ValueError(msg)


@pytest.mark.parametrize(
    "model_cls",
    [
        FakeMessagesListChatModel,
        FakeListChatModel,
        FakeChatModel,
        GenericFakeChatModel,
        ParrotFakeChatModel,
    ],
)
def test_fake_chat_model_bind_tools(model_cls: type) -> None:
    """All fake chat models should support bind_tools without raising."""
    model = _make_fake_model(model_cls)
    bound = model.bind_tools([Person])
    # bind_tools should return a Runnable (RunnableBinding)
    assert hasattr(bound, "invoke")


@pytest.mark.parametrize(
    "model_cls",
    [
        FakeMessagesListChatModel,
        FakeListChatModel,
        FakeChatModel,
        GenericFakeChatModel,
        ParrotFakeChatModel,
    ],
)
def test_fake_chat_model_with_structured_output(model_cls: type) -> None:
    """All fake chat models should support with_structured_output without raising."""
    model = _make_fake_model(model_cls)
    structured = model.with_structured_output(Person)
    # with_structured_output should return a Runnable chain
    assert hasattr(structured, "invoke")


def test_generic_fake_chat_model_with_structured_output_end_to_end() -> None:
    """Test that with_structured_output works end-to-end with tool_calls."""
    response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "Person",
                "args": {"name": "Alice", "age": 30},
                "id": "call_1",
            }
        ],
    )
    model = GenericFakeChatModel(messages=iter([response]))
    structured = model.with_structured_output(Person)
    result = structured.invoke("Who is Alice?")
    assert isinstance(result, Person)
    assert result.name == "Alice"
    assert result.age == 30


def test_fake_messages_list_with_structured_output_end_to_end() -> None:
    """Test that with_structured_output works end-to-end with tool_calls."""
    response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "Person",
                "args": {"name": "Bob", "age": 25},
                "id": "call_2",
            }
        ],
    )
    model = FakeMessagesListChatModel(responses=[response])
    structured = model.with_structured_output(Person)
    result = structured.invoke("Who is Bob?")
    assert isinstance(result, Person)
    assert result.name == "Bob"
    assert result.age == 25


def test_with_structured_output_include_raw() -> None:
    """Test with_structured_output with include_raw=True."""
    response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "Person",
                "args": {"name": "Charlie", "age": 35},
                "id": "call_3",
            }
        ],
    )
    model = GenericFakeChatModel(messages=iter([response]))
    structured = model.with_structured_output(Person, include_raw=True)
    result = structured.invoke("Who is Charlie?")
    assert isinstance(result, dict)
    assert isinstance(result["parsed"], Person)
    assert result["parsed"].name == "Charlie"
    assert result["parsing_error"] is None
    assert isinstance(result["raw"], AIMessage)
