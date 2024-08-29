"""Tests for verifying that testing utility code works as expected."""

from itertools import cycle
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

import pytest

from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.language_models import (
    FakeMessagesListChatModel,
    GenericFakeChatModel,
    ParrotFakeChatModel,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
)
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk
from tests.unit_tests.stubs import (
    AnyStr,
    _AnyIdAIMessage,
    _AnyIdAIMessageChunk,
    _AnyIdHumanMessage,
)


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
        ]
    )
    model = GenericFakeChatModel(messages=infinite_cycle)
    chunks = [chunk async for chunk in model.astream("meow")]
    assert chunks == [
        _AnyIdAIMessageChunk(content="hello"),
        _AnyIdAIMessageChunk(content=" "),
        _AnyIdAIMessageChunk(content="goodbye"),
    ]
    assert len({chunk.id for chunk in chunks}) == 1

    chunks = [chunk for chunk in model.stream("meow")]
    assert chunks == [
        _AnyIdAIMessageChunk(content="hello"),
        _AnyIdAIMessageChunk(content=" "),
        _AnyIdAIMessageChunk(content="goodbye"),
    ]
    assert len({chunk.id for chunk in chunks}) == 1

    # Test streaming of additional kwargs.
    # Relying on insertion order of the additional kwargs dict
    message = AIMessage(content="", additional_kwargs={"foo": 42, "bar": 24})
    model = GenericFakeChatModel(messages=cycle([message]))
    chunks = [chunk async for chunk in model.astream("meow")]
    assert chunks == [
        AIMessageChunk(content="", additional_kwargs={"foo": 42}, id=AnyStr()),
        AIMessageChunk(content="", additional_kwargs={"bar": 24}, id=AnyStr()),
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
        AIMessageChunk(
            content="",
            additional_kwargs={"function_call": {"name": "move_file"}},
            id=AnyStr(),
        ),
        AIMessageChunk(
            content="",
            additional_kwargs={
                "function_call": {"arguments": '{\n  "source_path": "foo"'},
            },
            id=AnyStr(),
        ),
        AIMessageChunk(
            content="",
            additional_kwargs={"function_call": {"arguments": ","}},
            id=AnyStr(),
        ),
        AIMessageChunk(
            content="",
            additional_kwargs={
                "function_call": {"arguments": '\n  "destination_path": "bar"\n}'},
            },
            id=AnyStr(),
        ),
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
        _AnyIdAIMessageChunk(content="goodbye"),
    ]
    assert len({chunk.id for chunk in final.state["streamed_output"]}) == 1


async def test_callback_handlers() -> None:
    """Verify that model is implemented correctly with handlers working."""

    class MyCustomAsyncHandler(AsyncCallbackHandler):
        def __init__(self, store: List[str]) -> None:
            self.store = store

        async def on_chat_model_start(
            self,
            serialized: Dict[str, Any],
            messages: List[List[BaseMessage]],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
        ) -> Any:
            # Do nothing
            # Required to implement since this is an abstract method
            pass

        async def on_llm_new_token(
            self,
            token: str,
            *,
            chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            **kwargs: Any,
        ) -> None:
            self.store.append(token)

    infinite_cycle = cycle(
        [
            AIMessage(content="hello goodbye"),
        ]
    )
    model = GenericFakeChatModel(messages=infinite_cycle)
    tokens: List[str] = []
    # New model
    results = list(model.stream("meow", {"callbacks": [MyCustomAsyncHandler(tokens)]}))
    assert results == [
        _AnyIdAIMessageChunk(content="hello"),
        _AnyIdAIMessageChunk(content=" "),
        _AnyIdAIMessageChunk(content="goodbye"),
    ]
    assert tokens == ["hello", " ", "goodbye"]
    assert len({chunk.id for chunk in results}) == 1


def test_chat_model_inputs() -> None:
    fake = ParrotFakeChatModel()

    assert fake.invoke("hello") == _AnyIdHumanMessage(content="hello")
    assert fake.invoke([("ai", "blah")]) == _AnyIdAIMessage(content="blah")
    assert fake.invoke([AIMessage(content="blah")]) == _AnyIdAIMessage(content="blah")


@pytest.mark.parametrize(
    "responses, expected_contents",
    [
        (
            [AIMessageChunk(content="Hello"), AIMessageChunk(content="Bye")],
            ["Hello", "Bye", "Hello"],
        ),
        (["Hello", "Bye"], ["Hello", "Bye", "Hello"]),
    ],
)
async def test_fake_messages_list_chat_model_invoke(
    responses: List[Union[BaseMessage, BaseMessageChunk, str]],
    expected_contents: List[str],
) -> None:
    model = FakeMessagesListChatModel(responses=responses)
    # invoke
    for expected in expected_contents:
        result = model.invoke("Test input")
        assert isinstance(result, AIMessage)
        assert result.content == expected
    # reset the model
    model.i = 0
    # ainvoke
    for expected in expected_contents:
        result = await model.ainvoke("Test input")
        assert isinstance(result, AIMessage)
        assert result.content == expected
    # reset the model
    model.i = 0
    # stream
    for expected in expected_contents:
        chunks = list(model.stream("Test input"))
        assert isinstance(chunks[0], AIMessageChunk)
        assert len(chunks) > 1
        full = chunks[0]
        for chunk in chunks[1:]:
            full += chunk
        assert full.content == expected
    # reset the model
    model.i = 0
    # astream
    for expected in expected_contents:
        chunks = [chunk async for chunk in model.astream("Test input")]
        assert isinstance(chunks[0], AIMessageChunk)
        assert len(chunks) > 1
        full = chunks[0]
        for chunk in chunks[1:]:
            full += chunk
        assert full.content == expected


@pytest.mark.parametrize(
    "responses, expected_chunks",
    [
        ([["Hello", "World"]], ["Hello", "World"]),
        (
            [[AIMessageChunk(content="Hello"), AIMessageChunk(content="World")]],
            ["Hello", "World"],
        ),
    ],
)
def test_fake_messages_list_chat_model_stream(
    responses: List[List[Union[BaseMessage, BaseMessageChunk, str]]],
    expected_chunks: List[str],
) -> None:
    model = FakeMessagesListChatModel(responses=responses)
    chunks = list(model.stream("Test input"))
    assert len(chunks) == len(expected_chunks)
    for chunk, expected_content in zip(chunks, expected_chunks):
        assert isinstance(chunk, AIMessageChunk)
        assert chunk.content == expected_content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "responses, expected_chunks",
    [
        ([["Hello", "World"]], ["Hello", "World"]),
        (
            [[AIMessageChunk(content="Hello"), AIMessageChunk(content="World")]],
            ["Hello", "World"],
        ),
    ],
)
async def test_fake_messages_list_chat_model_astream(
    responses: List[List[Union[BaseMessage, BaseMessageChunk, str]]],
    expected_chunks: List[str],
) -> None:
    model = FakeMessagesListChatModel(responses=responses)
    chunks = [chunk async for chunk in model.astream("Test input")]
    assert len(chunks) == len(expected_chunks)
    for chunk, expected_chunk in zip(chunks, expected_chunks):
        assert isinstance(chunk, AIMessageChunk)
        assert chunk.content == expected_chunk
