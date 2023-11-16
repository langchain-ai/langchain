from typing import Callable

from langchain.memory import ChatMessageHistory
from langchain.schema import AIMessage, HumanMessage
from langchain.schema.runnable import RunnableConfig, RunnableLambda


def get_factory() -> Callable:
    chat_history_store = {}

    def history_factory(session_id: str) -> ChatMessageHistory:
        if session_id not in chat_history_store:
            chat_history_store[session_id] = ChatMessageHistory()
        return chat_history_store[session_id]

    return history_factory


def test_input_messages() -> None:
    runnable = RunnableLambda(
        lambda messages: "you said: "
        + "\n".join(str(m.content) for m in messages if isinstance(m, HumanMessage))
    )
    factory = get_factory()
    with_history = runnable.with_message_history(factory)
    config: RunnableConfig = {"configurable": {"thread_id": "foo"}}
    output = with_history.invoke(HumanMessage(content="hello"), config)
    assert output == "you said: hello"
    output = with_history.invoke(HumanMessage(content="good bye"), config)
    assert output == "you said: hello\ngood bye"


def test_input_dict() -> None:
    runnable = RunnableLambda(
        lambda input: "you said: "
        + "\n".join(
            str(m.content) for m in input["messages"] if isinstance(m, HumanMessage)
        )
    )
    factory = get_factory()
    with_history = runnable.with_message_history(factory, input_messages_key="messages")
    config: RunnableConfig = {"configurable": {"thread_id": "foo"}}
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
    factory = get_factory()
    with_history = runnable.with_message_history(
        factory, input_messages_key="input", message_history_key="history"
    )
    config: RunnableConfig = {"configurable": {"thread_id": "foo"}}
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
    factory = get_factory()
    with_history = runnable.with_message_history(
        factory, input_messages_key="input", message_history_key="history"
    )
    config: RunnableConfig = {"configurable": {"thread_id": "foo"}}
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
    factory = get_factory()
    with_history = runnable.with_message_history(
        factory, input_messages_key="input", message_history_key="history"
    )
    config: RunnableConfig = {"configurable": {"thread_id": "foo"}}
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
    factory = get_factory()
    with_history = runnable.with_message_history(
        factory, input_messages_key="input", message_history_key="history"
    )
    config: RunnableConfig = {"configurable": {"thread_id": "foo"}}
    output = with_history.invoke({"input": "hello"}, config)
    assert output == {"output": [AIMessage(content="you said: hello")]}
    output = with_history.invoke({"input": "good bye"}, config)
    assert output == {"output": [AIMessage(content="you said: hello\ngood bye")]}


def test_get_input_schema() -> None:
    pass
