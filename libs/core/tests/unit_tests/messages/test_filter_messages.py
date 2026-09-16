from typing import List

import pytest

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
)


def test_filter_messages_by_include_types() -> None:
    messages: List[BaseMessage] = [
        SystemMessage(content="system prompt"),
        HumanMessage(content="hello"),
        AIMessage(content="hi there"),
        ToolMessage(content="tool output", tool_call_id="call_1"),
    ]

    filtered_ai = list(filter_messages(messages, include_types=[AIMessage]))
    assert len(filtered_ai) == 1
    assert filtered_ai[0].content == "hi there"

    filtered_human_sys = list(filter_messages(messages, include_types=["human", "system"]))
    assert len(filtered_human_sys) == 2
    assert [m.content for m in filtered_human_sys] == ["system prompt", "hello"]


def test_filter_messages_by_exclude_types() -> None:
    messages: List[BaseMessage] = [
        SystemMessage(content="system prompt"),
        HumanMessage(content="user query"),
        AIMessage(content="bot response"),
        FunctionMessage(name="func", content="result"),
    ]

    filtered = list(filter_messages(messages, exclude_types=["system", "function"]))
    assert len(filtered) == 2
    assert [m.content for m in filtered] == ["user query", "bot response"]


def test_filter_messages_by_include_names() -> None:
    messages: List[BaseMessage] = [
        HumanMessage(content="from alice", name="alice"),
        HumanMessage(content="from bob", name="bob"),
        AIMessage(content="from assistant", name="assistant_bot"),
    ]

    filtered = list(filter_messages(messages, include_names=["alice", "assistant_bot"]))
    assert len(filtered) == 2
    assert [m.content for m in filtered] == ["from alice", "from assistant"]


def test_filter_messages_by_exclude_names() -> None:
    messages: List[BaseMessage] = [
        HumanMessage(content="from alice", name="alice"),
        HumanMessage(content="from bob", name="bob"),
        ChatMessage(role="custom", content="from charlie", name="charlie"),
    ]

    filtered = list(filter_messages(messages, exclude_names=["bob"]))
    assert len(filtered) == 2
    assert [m.content for m in filtered] == ["from alice", "from charlie"]


def test_filter_messages_by_include_ids() -> None:
    messages: List[BaseMessage] = [
        HumanMessage(content="first", id="id-1"),
        AIMessage(content="second", id="id-2"),
        HumanMessage(content="third", id="id-3"),
    ]

    filtered = list(filter_messages(messages, include_ids=["id-1", "id-3"]))
    assert len(filtered) == 2
    assert [m.content for m in filtered] == ["first", "third"]


def test_filter_messages_empty_input() -> None:
    filtered = list(filter_messages([], include_types=["human"]))
    assert filtered == []


def test_filter_messages_exclude_all() -> None:
    messages: List[BaseMessage] = [
        HumanMessage(content="msg1"),
        AIMessage(content="msg2"),
    ]
    filtered = list(filter_messages(messages, exclude_types=["human", "ai"]))
    assert filtered == []
