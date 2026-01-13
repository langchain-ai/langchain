"""Tests for verifying that testing utility code works as expected."""

from itertools import cycle
from typing import Any
from uuid import UUID

from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk
from typing_extensions import override

from tests.unit_tests.llms.fake_chat_model import GenericFakeChatModel
from tests.unit_tests.stubs import _AnyIdAIMessage, _AnyIdAIMessageChunk


def test_generic_fake_chat_model_invoke() -> None:
    # Will alternate between responding with hello and goodbye
    infinite_cycle = cycle([AIMessage(content="hello"), AIMessage(content="goodbye")])
    model = GenericFakeChatModel(messages=infinite_cycle)
    response = model.invoke("meow")
    assert response == _AnyIdAIMessage(content="hello")
    response = model.invoke("kitty")
    assert response == _AnyIdAIMessage(content="goodbye")
    response = model.invoke("meow")
    assert response == _AnyIdAIMessage(content="hello")


async def test_generic_fake_chat_model_ainvoke() -> None:
    # Will alternate between responding with hello and goodbye
    infinite_cycle = cycle([AIMessage(content="hello"), AIMessage(content="goodbye")])
    model = GenericFakeChatModel(messages=infinite_cycle)
    response = await model.ainvoke("meow")
    assert response == _AnyIdAIMessage(content="hello")
    response = await model.ainvoke("kitty")
    assert response == _AnyIdAIMessage(content="goodbye")
    response = await model.ainvoke("meow")
    assert response == _AnyIdAIMessage(content="hello")


async def test_generic_fake_chat_model_stream() -> None:
    """Test streaming."""
    infinite_cycle = cycle(
        [
            AIMessage(content="hello goodbye"),
        ],
    )
    model = GenericFakeChatModel(messages=infinite_cycle)
    chunks = [chunk async for chunk in model.astream("meow")]
    assert chunks == [
        _AnyIdAIMessageChunk(content="hello"),
        _AnyIdAIMessageChunk(content=" "),
        _AnyIdAIMessageChunk(content="goodbye", chunk_position="last"),
    ]

    chunks = list(model.stream("meow"))
    assert chunks == [
        _AnyIdAIMessageChunk(content="hello"),
        _AnyIdAIMessageChunk(content=" "),
        _AnyIdAIMessageChunk(content="goodbye", chunk_position="last"),
    ]

    # Test streaming of additional kwargs.
    # Relying on insertion order of the additional kwargs dict
    message = AIMessage(content="", additional_kwargs={"foo": 42, "bar": 24})
    model = GenericFakeChatModel(messages=cycle([message]))
    chunks = [chunk async for chunk in model.astream("meow")]
    assert chunks == [
        _AnyIdAIMessageChunk(content="", additional_kwargs={"foo": 42}),
        _AnyIdAIMessageChunk(content="", additional_kwargs={"bar": 24}),
        _AnyIdAIMessageChunk(content="", chunk_position="last"),
    ]

    message = AIMessage(
        id="a1",
        content="",
        additional_kwargs={
            "function_call": {
                "name": "move_file",
                "arguments": '{\n  "source_path": "foo",\n  "'
                'destination_path": "bar"\n}',
            },
        },
    )
    model = GenericFakeChatModel(messages=cycle([message]))
    chunks = [chunk async for chunk in model.astream("meow")]

    assert chunks == [
        AIMessageChunk(
            content="",
            additional_kwargs={"function_call": {"name": "move_file"}},
            id="a1",
        ),
        AIMessageChunk(
            id="a1",
            content="",
            additional_kwargs={
                "function_call": {"arguments": '{\n  "source_path": "foo"'},
            },
        ),
        AIMessageChunk(
            id="a1",
            content="",
            additional_kwargs={"function_call": {"arguments": ","}},
        ),
        AIMessageChunk(
            id="a1",
            content="",
            additional_kwargs={
                "function_call": {"arguments": '\n  "destination_path": "bar"\n}'},
            },
        ),
        _AnyIdAIMessageChunk(content="", chunk_position="last"),
    ]

    accumulate_chunks = None
    for chunk in chunks:
        if accumulate_chunks is None:
            accumulate_chunks = chunk
        else:
            accumulate_chunks += chunk

    assert accumulate_chunks == AIMessageChunk(
        id="a1",
        content="",
        additional_kwargs={
            "function_call": {
                "name": "move_file",
                "arguments": '{\n  "source_path": "foo",\n  "'
                'destination_path": "bar"\n}',
            },
        },
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
        _AnyIdAIMessageChunk(content="hello"),
        _AnyIdAIMessageChunk(content=" "),
        _AnyIdAIMessageChunk(content="goodbye", chunk_position="last"),
    ]


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
            token: str | list[str | dict[str, Any]],
            *,
            chunk: GenerationChunk | ChatGenerationChunk | None = None,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            tags: list[str] | None = None,
            **kwargs: Any,
        ) -> None:
            self.store.append(str(token))

    infinite_cycle = cycle(
        [
            AIMessage(content="hello goodbye"),
        ],
    )
    model = GenericFakeChatModel(messages=infinite_cycle)
    tokens: list[str] = []
    # New model
    results = [
        chunk
        async for chunk in model.astream(
            "meow",
            {"callbacks": [MyCustomAsyncHandler(tokens)]},
        )
    ]
    assert results == [
        _AnyIdAIMessageChunk(content="hello"),
        _AnyIdAIMessageChunk(content=" "),
        _AnyIdAIMessageChunk(content="goodbye", chunk_position="last"),
    ]
    assert tokens == ["hello", " ", "goodbye"]
