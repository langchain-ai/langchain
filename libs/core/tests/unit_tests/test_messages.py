import unittest
from typing import List

import pytest

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    ToolMessage,
    convert_to_messages,
    get_buffer_string,
    message_chunk_to_message,
    messages_from_dict,
    messages_to_dict,
)


def test_message_chunks() -> None:
    assert AIMessageChunk(content="I am") + AIMessageChunk(
        content=" indeed."
    ) == AIMessageChunk(
        content="I am indeed."
    ), "MessageChunk + MessageChunk should be a MessageChunk"

    assert (
        AIMessageChunk(content="I am") + HumanMessageChunk(content=" indeed.")
        == AIMessageChunk(content="I am indeed.")
    ), "MessageChunk + MessageChunk should be a MessageChunk of same class as the left side"  # noqa: E501

    assert (
        AIMessageChunk(content="", additional_kwargs={"foo": "bar"})
        + AIMessageChunk(content="", additional_kwargs={"baz": "foo"})
        == AIMessageChunk(content="", additional_kwargs={"foo": "bar", "baz": "foo"})
    ), "MessageChunk + MessageChunk should be a MessageChunk with merged additional_kwargs"  # noqa: E501

    assert (
        AIMessageChunk(
            content="", additional_kwargs={"function_call": {"name": "web_search"}}
        )
        + AIMessageChunk(
            content="", additional_kwargs={"function_call": {"arguments": None}}
        )
        + AIMessageChunk(
            content="", additional_kwargs={"function_call": {"arguments": "{\n"}}
        )
        + AIMessageChunk(
            content="",
            additional_kwargs={
                "function_call": {"arguments": '  "query": "turtles"\n}'}
            },
        )
        == AIMessageChunk(
            content="",
            additional_kwargs={
                "function_call": {
                    "name": "web_search",
                    "arguments": '{\n  "query": "turtles"\n}',
                }
            },
        )
    ), "MessageChunk + MessageChunk should be a MessageChunk with merged additional_kwargs"  # noqa: E501


def test_chat_message_chunks() -> None:
    assert ChatMessageChunk(role="User", content="I am") + ChatMessageChunk(
        role="User", content=" indeed."
    ) == ChatMessageChunk(
        role="User", content="I am indeed."
    ), "ChatMessageChunk + ChatMessageChunk should be a ChatMessageChunk"

    with pytest.raises(ValueError):
        ChatMessageChunk(role="User", content="I am") + ChatMessageChunk(
            role="Assistant", content=" indeed."
        )

    assert (
        ChatMessageChunk(role="User", content="I am")
        + AIMessageChunk(content=" indeed.")
        == ChatMessageChunk(role="User", content="I am indeed.")
    ), "ChatMessageChunk + other MessageChunk should be a ChatMessageChunk with the left side's role"  # noqa: E501

    assert AIMessageChunk(content="I am") + ChatMessageChunk(
        role="User", content=" indeed."
    ) == AIMessageChunk(
        content="I am indeed."
    ), "Other MessageChunk + ChatMessageChunk should be a MessageChunk as the left side"  # noqa: E501


def test_function_message_chunks() -> None:
    assert FunctionMessageChunk(name="hello", content="I am") + FunctionMessageChunk(
        name="hello", content=" indeed."
    ) == FunctionMessageChunk(
        name="hello", content="I am indeed."
    ), "FunctionMessageChunk + FunctionMessageChunk should be a FunctionMessageChunk"

    with pytest.raises(ValueError):
        FunctionMessageChunk(name="hello", content="I am") + FunctionMessageChunk(
            name="bye", content=" indeed."
        )


def test_ani_message_chunks() -> None:
    assert AIMessageChunk(example=True, content="I am") + AIMessageChunk(
        example=True, content=" indeed."
    ) == AIMessageChunk(
        example=True, content="I am indeed."
    ), "AIMessageChunk + AIMessageChunk should be a AIMessageChunk"

    with pytest.raises(ValueError):
        AIMessageChunk(example=True, content="I am") + AIMessageChunk(
            example=False, content=" indeed."
        )


class TestGetBufferString(unittest.TestCase):
    def setUp(self) -> None:
        self.human_msg = HumanMessage(content="human")
        self.ai_msg = AIMessage(content="ai")
        self.sys_msg = SystemMessage(content="system")
        self.func_msg = FunctionMessage(name="func", content="function")
        self.tool_msg = ToolMessage(tool_call_id="tool_id", content="tool")
        self.chat_msg = ChatMessage(role="Chat", content="chat")

    def test_empty_input(self) -> None:
        self.assertEqual(get_buffer_string([]), "")

    def test_valid_single_message(self) -> None:
        expected_output = f"Human: {self.human_msg.content}"
        self.assertEqual(
            get_buffer_string([self.human_msg]),
            expected_output,
        )

    def test_custom_human_prefix(self) -> None:
        prefix = "H"
        expected_output = f"{prefix}: {self.human_msg.content}"
        self.assertEqual(
            get_buffer_string([self.human_msg], human_prefix="H"),
            expected_output,
        )

    def test_custom_ai_prefix(self) -> None:
        prefix = "A"
        expected_output = f"{prefix}: {self.ai_msg.content}"
        self.assertEqual(
            get_buffer_string([self.ai_msg], ai_prefix="A"),
            expected_output,
        )

    def test_multiple_msg(self) -> None:
        msgs = [
            self.human_msg,
            self.ai_msg,
            self.sys_msg,
            self.func_msg,
            self.tool_msg,
            self.chat_msg,
        ]
        expected_output = "\n".join(
            [
                "Human: human",
                "AI: ai",
                "System: system",
                "Function: function",
                "Tool: tool",
                "Chat: chat",
            ]
        )
        self.assertEqual(
            get_buffer_string(msgs),
            expected_output,
        )


def test_multiple_msg() -> None:
    human_msg = HumanMessage(content="human", additional_kwargs={"key": "value"})
    ai_msg = AIMessage(content="ai")
    sys_msg = SystemMessage(content="sys")

    msgs = [
        human_msg,
        ai_msg,
        sys_msg,
    ]
    assert messages_from_dict(messages_to_dict(msgs)) == msgs


def test_message_chunk_to_message() -> None:
    assert message_chunk_to_message(
        AIMessageChunk(content="I am", additional_kwargs={"foo": "bar"})
    ) == AIMessage(content="I am", additional_kwargs={"foo": "bar"})
    assert message_chunk_to_message(HumanMessageChunk(content="I am")) == HumanMessage(
        content="I am"
    )
    assert message_chunk_to_message(
        ChatMessageChunk(role="User", content="I am")
    ) == ChatMessage(role="User", content="I am")
    assert message_chunk_to_message(
        FunctionMessageChunk(name="hello", content="I am")
    ) == FunctionMessage(name="hello", content="I am")


def test_tool_calls_merge() -> None:
    chunks: List[dict] = [
        dict(content=""),
        dict(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_CwGAsESnXehQEjiAIWzinlva",
                        "function": {"arguments": "", "name": "person"},
                        "type": "function",
                    }
                ]
            },
        ),
        dict(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "index": 0,
                        "id": None,
                        "function": {"arguments": '{"na', "name": None},
                        "type": None,
                    }
                ]
            },
        ),
        dict(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "index": 0,
                        "id": None,
                        "function": {"arguments": 'me": ', "name": None},
                        "type": None,
                    }
                ]
            },
        ),
        dict(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "index": 0,
                        "id": None,
                        "function": {"arguments": '"jane"', "name": None},
                        "type": None,
                    }
                ]
            },
        ),
        dict(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "index": 0,
                        "id": None,
                        "function": {"arguments": ', "a', "name": None},
                        "type": None,
                    }
                ]
            },
        ),
        dict(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "index": 0,
                        "id": None,
                        "function": {"arguments": 'ge": ', "name": None},
                        "type": None,
                    }
                ]
            },
        ),
        dict(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "index": 0,
                        "id": None,
                        "function": {"arguments": "2}", "name": None},
                        "type": None,
                    }
                ]
            },
        ),
        dict(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "index": 1,
                        "id": "call_zXSIylHvc5x3JUAPcHZR5GZI",
                        "function": {"arguments": "", "name": "person"},
                        "type": "function",
                    }
                ]
            },
        ),
        dict(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "index": 1,
                        "id": None,
                        "function": {"arguments": '{"na', "name": None},
                        "type": None,
                    }
                ]
            },
        ),
        dict(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "index": 1,
                        "id": None,
                        "function": {"arguments": 'me": ', "name": None},
                        "type": None,
                    }
                ]
            },
        ),
        dict(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "index": 1,
                        "id": None,
                        "function": {"arguments": '"bob",', "name": None},
                        "type": None,
                    }
                ]
            },
        ),
        dict(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "index": 1,
                        "id": None,
                        "function": {"arguments": ' "ag', "name": None},
                        "type": None,
                    }
                ]
            },
        ),
        dict(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "index": 1,
                        "id": None,
                        "function": {"arguments": 'e": 3', "name": None},
                        "type": None,
                    }
                ]
            },
        ),
        dict(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "index": 1,
                        "id": None,
                        "function": {"arguments": "}", "name": None},
                        "type": None,
                    }
                ]
            },
        ),
        dict(content=""),
    ]

    final = None

    for chunk in chunks:
        msg = AIMessageChunk(**chunk)
        if final is None:
            final = msg
        else:
            final = final + msg

    assert final == AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_CwGAsESnXehQEjiAIWzinlva",
                    "function": {
                        "arguments": '{"name": "jane", "age": 2}',
                        "name": "person",
                    },
                    "type": "function",
                },
                {
                    "index": 1,
                    "id": "call_zXSIylHvc5x3JUAPcHZR5GZI",
                    "function": {
                        "arguments": '{"name": "bob", "age": 3}',
                        "name": "person",
                    },
                    "type": "function",
                },
            ]
        },
    )


def test_convert_to_messages() -> None:
    # dicts
    assert convert_to_messages(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "ai", "content": "Hi!"},
            {"role": "human", "content": "Hello!", "name": "Jane"},
            {
                "role": "assistant",
                "content": "Hi!",
                "name": "JaneBot",
                "function_call": {"name": "greet", "arguments": '{"name": "Jane"}'},
            },
            {"role": "function", "name": "greet", "content": "Hi!"},
            {"role": "tool", "tool_call_id": "tool_id", "content": "Hi!"},
        ]
    ) == [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hello!"),
        AIMessage(content="Hi!"),
        HumanMessage(content="Hello!", name="Jane"),
        AIMessage(
            content="Hi!",
            name="JaneBot",
            additional_kwargs={
                "function_call": {"name": "greet", "arguments": '{"name": "Jane"}'}
            },
        ),
        FunctionMessage(name="greet", content="Hi!"),
        ToolMessage(tool_call_id="tool_id", content="Hi!"),
    ]

    # tuples
    assert convert_to_messages(
        [
            ("system", "You are a helpful assistant."),
            "hello!",
            ("ai", "Hi!"),
            ("human", "Hello!"),
            ("assistant", "Hi!"),
        ]
    ) == [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="hello!"),
        AIMessage(content="Hi!"),
        HumanMessage(content="Hello!"),
        AIMessage(content="Hi!"),
    ]
