import unittest
import uuid
from typing import Optional, Union

import pytest

from langchain_core.documents import Document
from langchain_core.load import dumpd, load
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
    convert_to_messages,
    convert_to_openai_image_block,
    get_buffer_string,
    is_data_content_block,
    merge_content,
    message_chunk_to_message,
    message_to_dict,
    messages_from_dict,
    messages_to_dict,
)
from langchain_core.messages.tool import invalid_tool_call as create_invalid_tool_call
from langchain_core.messages.tool import tool_call as create_tool_call
from langchain_core.messages.tool import tool_call_chunk as create_tool_call_chunk
from langchain_core.utils._merge import merge_lists


def test_message_init() -> None:
    for doc in [
        BaseMessage(type="foo", content="bar"),
        BaseMessage(type="foo", content="bar", id=None),
        BaseMessage(type="foo", content="bar", id="1"),
        BaseMessage(type="foo", content="bar", id=1),
    ]:
        assert isinstance(doc, BaseMessage)


def test_message_chunks() -> None:
    assert AIMessageChunk(content="I am", id="ai3") + AIMessageChunk(
        content=" indeed."
    ) == AIMessageChunk(content="I am indeed.", id="ai3"), (
        "MessageChunk + MessageChunk should be a MessageChunk"
    )

    assert AIMessageChunk(content="I am", id="ai2") + HumanMessageChunk(
        content=" indeed.", id="human1"
    ) == AIMessageChunk(content="I am indeed.", id="ai2"), (
        "MessageChunk + MessageChunk should be a MessageChunk "
        "of same class as the left side"
    )

    assert AIMessageChunk(
        content="", additional_kwargs={"foo": "bar"}
    ) + AIMessageChunk(content="", additional_kwargs={"baz": "foo"}) == AIMessageChunk(
        content="", additional_kwargs={"foo": "bar", "baz": "foo"}
    ), (
        "MessageChunk + MessageChunk should be a MessageChunk "
        "with merged additional_kwargs"
    )

    assert AIMessageChunk(
        content="", additional_kwargs={"function_call": {"name": "web_search"}}
    ) + AIMessageChunk(
        content="", additional_kwargs={"function_call": {"arguments": None}}
    ) + AIMessageChunk(
        content="", additional_kwargs={"function_call": {"arguments": "{\n"}}
    ) + AIMessageChunk(
        content="",
        additional_kwargs={"function_call": {"arguments": '  "query": "turtles"\n}'}},
    ) == AIMessageChunk(
        content="",
        additional_kwargs={
            "function_call": {
                "name": "web_search",
                "arguments": '{\n  "query": "turtles"\n}',
            }
        },
    ), (
        "MessageChunk + MessageChunk should be a MessageChunk "
        "with merged additional_kwargs"
    )

    # Test tool calls
    assert (
        AIMessageChunk(
            content="",
            tool_call_chunks=[
                create_tool_call_chunk(name="tool1", args="", id="1", index=0)
            ],
