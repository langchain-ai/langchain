from collections import defaultdict
from typing import DefaultDict

import pytest
from freezegun import freeze_time
from pytest_mock import MockerFixture

from langchain.chat_models.fake import FakeListChatModel
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.prompts.chat import (
    ChatPromptValue,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.schema.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)
from tests.unit_tests.schema.runnable.test_runnable import FakeTracer


@freeze_time("2023-01-01")
def test_prompt_with_chat_model(mocker: MockerFixture) -> None:
    histories: DefaultDict[str, ChatMessageHistory] = defaultdict(ChatMessageHistory)

    def history_factory(session_id: str) -> ChatMessageHistory:
        return histories[session_id]

    prompt = (
        SystemMessagePromptTemplate.from_template("You are a nice assistant.")
        + MessagesPlaceholder(variable_name="history")
        + "{question}"
    )
    chat = FakeListChatModel(responses=["foo", "bar", "baz"])

    chain = (prompt | chat).with_message_history(history_factory, "question")

    # Test invoke
    prompt_spy = mocker.spy(prompt.__class__, "invoke")
    chat_spy = mocker.spy(chat.__class__, "invoke")
    tracer = FakeTracer()
    assert chain.invoke(
        {"question": "What is your name?"},
        {"callbacks": [tracer], "configurable": {"session_id": "123"}},
    ) == AIMessage(content="foo")
    assert prompt_spy.call_args.args[1] == {
        "history": [],
        "question": "What is your name?",
    }
    assert chat_spy.call_args.args[1] == ChatPromptValue(
        messages=[
            SystemMessage(content="You are a nice assistant."),
            HumanMessage(content="What is your name?"),
        ]
    )

    mocker.stop(prompt_spy)
    mocker.stop(chat_spy)

    # Test stream
    prompt_spy = mocker.spy(prompt.__class__, "invoke")
    chat_spy = mocker.spy(chat.__class__, "stream")
    tracer = FakeTracer()
    assert [
        *chain.stream(
            {"question": "What is your name?"},
            {"callbacks": [tracer], "configurable": {"session_id": "123"}},
        )
    ] == [
        AIMessageChunk(content="b"),
        AIMessageChunk(content="a"),
        AIMessageChunk(content="r"),
    ]
    assert prompt_spy.call_args.args[1] == {
        "question": "What is your name?",
        "history": [
            HumanMessage(content="What is your name?"),
            AIMessage(content="foo"),
        ],
    }
    assert chat_spy.call_args.args[1] == ChatPromptValue(
        messages=[
            SystemMessage(content="You are a nice assistant."),
            HumanMessage(content="What is your name?"),
            AIMessage(content="foo"),
            HumanMessage(content="What is your name?"),
        ]
    )


@pytest.mark.asyncio
@freeze_time("2023-01-01")
async def test_prompt_with_chat_model_async(mocker: MockerFixture) -> None:
    histories: DefaultDict[str, ChatMessageHistory] = defaultdict(ChatMessageHistory)

    def history_factory(session_id: str) -> ChatMessageHistory:
        return histories[session_id]

    prompt = (
        SystemMessagePromptTemplate.from_template("You are a nice assistant.")
        + MessagesPlaceholder(variable_name="history")
        + "{question}"
    )
    chat = FakeListChatModel(responses=["foo", "bar", "baz"])

    chain = (prompt | chat).with_message_history(history_factory, "question")

    # Test invoke
    prompt_spy = mocker.spy(prompt.__class__, "ainvoke")
    chat_spy = mocker.spy(chat.__class__, "ainvoke")
    tracer = FakeTracer()
    assert await chain.ainvoke(
        {"question": "What is your name?"},
        {"callbacks": [tracer], "configurable": {"session_id": "123"}},
    ) == AIMessage(content="foo")
    assert prompt_spy.call_args.args[1] == {
        "history": [],
        "question": "What is your name?",
    }
    assert chat_spy.call_args.args[1] == ChatPromptValue(
        messages=[
            SystemMessage(content="You are a nice assistant."),
            HumanMessage(content="What is your name?"),
        ]
    )

    mocker.stop(prompt_spy)
    mocker.stop(chat_spy)

    # Test stream
    prompt_spy = mocker.spy(prompt.__class__, "ainvoke")
    chat_spy = mocker.spy(chat.__class__, "astream")
    tracer = FakeTracer()
    assert [
        a
        async for a in chain.astream(
            {"question": "What is your name?"},
            {"callbacks": [tracer], "configurable": {"session_id": "123"}},
        )
    ] == [
        AIMessageChunk(content="b"),
        AIMessageChunk(content="a"),
        AIMessageChunk(content="r"),
    ]
    assert prompt_spy.call_args.args[1] == {
        "question": "What is your name?",
        "history": [
            HumanMessage(content="What is your name?"),
            AIMessage(content="foo"),
        ],
    }
    assert chat_spy.call_args.args[1] == ChatPromptValue(
        messages=[
            SystemMessage(content="You are a nice assistant."),
            HumanMessage(content="What is your name?"),
            AIMessage(content="foo"),
            HumanMessage(content="What is your name?"),
        ]
    )
