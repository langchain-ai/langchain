"""Tests for verifying that testing utility code works as expected."""

import time
from itertools import cycle
from typing import Any, Optional, Union, cast
from uuid import UUID

from typing_extensions import override

from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.language_models import (
    FakeListChatModel,
    FakeMessagesListChatModel,
    GenericFakeChatModel,
    ParrotFakeChatModel,
)
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
    assert response == _any_id_ai_message(content="hello", additional_kwargs={})
    response = model.invoke("kitty")
    assert response == _any_id_ai_message(content="goodbye", additional_kwargs={})
    response = model.invoke("meow")
    assert response == _any_id_ai_message(content="hello", additional_kwargs={})


async def test_generic_fake_chat_model_ainvoke() -> None:
    # Will alternate between responding with hello and goodbye
    infinite_cycle = cycle([AIMessage(content="hello"), AIMessage(content="goodbye")])
    model = GenericFakeChatModel(messages=infinite_cycle)
    response = await model.ainvoke("meow")
    assert response == _any_id_ai_message(content="hello", additional_kwargs={})
    response = await model.ainvoke("kitty")
    assert response == _any_id_ai_message(content="goodbye", additional_kwargs={})
    response = await model.ainvoke("meow")
    assert response == _any_id_ai_message(content="hello", additional_kwargs={})


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
        _any_id_ai_message_chunk(content="hello", additional_kwargs={}),
        _any_id_ai_message_chunk(content=" ", additional_kwargs={}),
        _any_id_ai_message_chunk(
            content="goodbye",
            chunk_position="last",
            additional_kwargs={},
        ),
    ]
    assert len({chunk.id for chunk in chunks}) == 1

    chunks = list(model.stream("meow"))
    assert chunks == [
        _any_id_ai_message_chunk(content="hello", additional_kwargs={}),
        _any_id_ai_message_chunk(content=" ", additional_kwargs={}),
        _any_id_ai_message_chunk(
            content="goodbye",
            chunk_position="last",
            additional_kwargs={},
        ),
    ]
    assert len({chunk.id for chunk in chunks}) == 1

    # Test streaming of additional kwargs.
    # Relying on insertion order of the additional kwargs dict
    message = AIMessage(content="", additional_kwargs={"foo": 42, "bar": 24})
    model = GenericFakeChatModel(messages=cycle([message]))
    chunks = [chunk async for chunk in model.astream("meow")]
    assert chunks == [
        _any_id_ai_message_chunk(
            content="", additional_kwargs={"foo": 42, "output_version": "v0"}
        ),
        _any_id_ai_message_chunk(
            content="", additional_kwargs={"bar": 24, "output_version": "v0"}
        ),
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
            additional_kwargs={
                "function_call": {"name": "move_file"},
                "output_version": "v0",
            },
        ),
        _any_id_ai_message_chunk(
            content="",
            additional_kwargs={
                "function_call": {"arguments": '{\n  "source_path": "foo"'},
                "output_version": "v0",
            },
        ),
        _any_id_ai_message_chunk(
            content="",
            additional_kwargs={
                "function_call": {"arguments": ","},
                "output_version": "v0",
            },
        ),
        _any_id_ai_message_chunk(
            content="",
            additional_kwargs={
                "function_call": {"arguments": '\n  "destination_path": "bar"\n}'},
                "output_version": "v0",
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
            },
            "output_version": "v0",
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
        _any_id_ai_message_chunk(content="hello", additional_kwargs={}),
        _any_id_ai_message_chunk(content=" ", additional_kwargs={}),
        _any_id_ai_message_chunk(
            content="goodbye",
            chunk_position="last",
            additional_kwargs={},
        ),
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
            parent_run_id: Optional[UUID] = None,
            tags: Optional[list[str]] = None,
            metadata: Optional[dict[str, Any]] = None,
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
            chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[list[str]] = None,
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
        _any_id_ai_message_chunk(content="hello", additional_kwargs={}),
        _any_id_ai_message_chunk(content=" ", additional_kwargs={}),
        _any_id_ai_message_chunk(
            content="goodbye",
            chunk_position="last",
            additional_kwargs={},
        ),
    ]
    assert tokens == ["hello", " ", "goodbye"]
    assert len({chunk.id for chunk in results}) == 1


def test_chat_model_inputs() -> None:
    fake = ParrotFakeChatModel()

    assert cast("HumanMessage", fake.invoke("hello")) == _any_id_human_message(
        content="hello"
    )
    assert fake.invoke([("ai", "blah")]) == _any_id_ai_message(
        content="blah", additional_kwargs={}
    )
    assert fake.invoke([AIMessage(content="blah")]) == _any_id_ai_message(
        content="blah", additional_kwargs={}
    )


def test_fake_list_chat_model_batch() -> None:
    expected = [
        _any_id_ai_message(content="a", additional_kwargs={}),
        _any_id_ai_message(content="b", additional_kwargs={}),
        _any_id_ai_message(content="c", additional_kwargs={}),
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
