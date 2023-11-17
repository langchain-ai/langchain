import datetime
from typing import Callable, Optional, Sequence

from langchain.memory import ChatMessageHistory
from langchain.pydantic_v1 import BaseModel
from langchain.schema import AIMessage, BaseMessage, HumanMessage
from langchain.schema.runnable import RunnableConfig, RunnableLambda


def _get_get_session_history() -> Callable:
    chat_history_store = {}

    def get_session_history(
        thread_id: str, *, user_id: Optional[str] = None
    ) -> ChatMessageHistory:
        session_id = thread_id if user_id is None else f"{user_id}:{thread_id}"
        if session_id not in chat_history_store:
            chat_history_store[session_id] = ChatMessageHistory()
        return chat_history_store[session_id]

    return get_session_history


def test_input_messages() -> None:
    runnable = RunnableLambda(
        lambda messages: "you said: "
        + "\n".join(str(m.content) for m in messages if isinstance(m, HumanMessage))
    )
    get_session_history = _get_get_session_history()
    with_history = runnable.with_message_history(get_session_history)
    config: RunnableConfig = {"configurable": {"thread_id": datetime.datetime.now()}}
    output = with_history.invoke([HumanMessage(content="hello")], config)
    assert output == "you said: hello"
    output = with_history.invoke([HumanMessage(content="good bye")], config)
    assert output == "you said: hello\ngood bye"


def test_input_dict() -> None:
    runnable = RunnableLambda(
        lambda input: "you said: "
        + "\n".join(
            str(m.content) for m in input["messages"] if isinstance(m, HumanMessage)
        )
    )
    get_session_history = _get_get_session_history()
    with_history = runnable.with_message_history(
        get_session_history, input_messages_key="messages"
    )
    config: RunnableConfig = {"configurable": {"thread_id": datetime.datetime.now()}}
    output = with_history.invoke({"messages": [HumanMessage(content="hello")]}, config)
    assert output == "you said: hello"
    output = with_history.invoke(
        {"messages": [HumanMessage(content="good bye")]}, config
    )
    assert output == "you said: hello\ngood bye"


def test_input_dict_with_history_key() -> None:
    runnable = RunnableLambda(
        lambda input: "you said: "
        + "\n".join(
            [str(m.content) for m in input["history"] if isinstance(m, HumanMessage)]
            + [input["input"]]
        )
    )
    get_session_history = _get_get_session_history()
    with_history = runnable.with_message_history(
        get_session_history, input_messages_key="input", history_messages_key="history"
    )
    config: RunnableConfig = {"configurable": {"thread_id": datetime.datetime.now()}}
    output = with_history.invoke({"input": "hello"}, config)
    assert output == "you said: hello"
    output = with_history.invoke({"input": "good bye"}, config)
    assert output == "you said: hello\ngood bye"


def test_output_message() -> None:
    runnable = RunnableLambda(
        lambda input: AIMessage(
            content="you said: "
            + "\n".join(
                [
                    str(m.content)
                    for m in input["history"]
                    if isinstance(m, HumanMessage)
                ]
                + [input["input"]]
            )
        )
    )
    get_session_history = _get_get_session_history()
    with_history = runnable.with_message_history(
        get_session_history, input_messages_key="input", history_messages_key="history"
    )
    config: RunnableConfig = {"configurable": {"thread_id": datetime.datetime.now()}}
    output = with_history.invoke({"input": "hello"}, config)
    assert output == AIMessage(content="you said: hello")
    output = with_history.invoke({"input": "good bye"}, config)
    assert output == AIMessage(content="you said: hello\ngood bye")


def test_output_messages() -> None:
    runnable = RunnableLambda(
        lambda input: [
            AIMessage(
                content="you said: "
                + "\n".join(
                    [
                        str(m.content)
                        for m in input["history"]
                        if isinstance(m, HumanMessage)
                    ]
                    + [input["input"]]
                )
            )
        ]
    )
    get_session_history = _get_get_session_history()
    with_history = runnable.with_message_history(
        get_session_history, input_messages_key="input", history_messages_key="history"
    )
    config: RunnableConfig = {"configurable": {"thread_id": datetime.datetime.now()}}
    output = with_history.invoke({"input": "hello"}, config)
    assert output == [AIMessage(content="you said: hello")]
    output = with_history.invoke({"input": "good bye"}, config)
    assert output == [AIMessage(content="you said: hello\ngood bye")]


def test_output_dict() -> None:
    runnable = RunnableLambda(
        lambda input: {
            "output": [
                AIMessage(
                    content="you said: "
                    + "\n".join(
                        [
                            str(m.content)
                            for m in input["history"]
                            if isinstance(m, HumanMessage)
                        ]
                        + [input["input"]]
                    )
                )
            ]
        }
    )
    get_session_history = _get_get_session_history()
    with_history = runnable.with_message_history(
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="output",
    )
    config: RunnableConfig = {"configurable": {"thread_id": datetime.datetime.now()}}
    output = with_history.invoke({"input": "hello"}, config)
    assert output == {"output": [AIMessage(content="you said: hello")]}
    output = with_history.invoke({"input": "good bye"}, config)
    assert output == {"output": [AIMessage(content="you said: hello\ngood bye")]}


def test_get_input_schema_input_dict() -> None:
    class RunnableWithChatHistoryInput(BaseModel):
        input: str
        history: Sequence[BaseMessage]

    runnable = RunnableLambda(
        lambda input: {
            "output": [
                AIMessage(
                    content="you said: "
                    + "\n".join(
                        [
                            str(m.content)
                            for m in input["history"]
                            if isinstance(m, HumanMessage)
                        ]
                        + [input["input"]]
                    )
                )
            ]
        }
    )
    get_session_history = _get_get_session_history()
    with_history = runnable.with_message_history(
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="output",
    )
    assert (
        with_history.get_input_schema().schema()
        == RunnableWithChatHistoryInput.schema()
    )


def test_get_input_schema_input_messages() -> None:
    class RunnableWithChatHistoryInput(BaseModel):
        __root__: Sequence[BaseMessage]

    runnable = RunnableLambda(
        lambda messages: {
            "output": [
                AIMessage(
                    content="you said: "
                    + "\n".join(
                        [
                            str(m.content)
                            for m in messages
                            if isinstance(m, HumanMessage)
                        ]
                    )
                )
            ]
        }
    )
    get_session_history = _get_get_session_history()
    with_history = runnable.with_message_history(
        get_session_history,
        output_messages_key="output",
    )
    assert (
        with_history.get_input_schema().schema()
        == RunnableWithChatHistoryInput.schema()
    )
