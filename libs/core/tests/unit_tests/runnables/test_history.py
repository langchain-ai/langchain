from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables.base import RunnableLambda
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.utils import ConfigurableFieldSpec
from tests.unit_tests.fake.memory import ChatMessageHistory


def test_interfaces() -> None:
    history = ChatMessageHistory()
    history.add_message(SystemMessage(content="system"))
    history.add_user_message("human 1")
    history.add_ai_message("ai")
    history.add_message(HumanMessage(content="human 2"))
    assert str(history) == "System: system\nHuman: human 1\nAI: ai\nHuman: human 2"


def _get_get_session_history(
    *, store: Optional[Dict[str, Any]] = None
) -> Callable[..., ChatMessageHistory]:
    chat_history_store = store if store is not None else {}

    def get_session_history(session_id: str, **kwargs: Any) -> ChatMessageHistory:
        if session_id not in chat_history_store:
            chat_history_store[session_id] = ChatMessageHistory()
        return chat_history_store[session_id]

    return get_session_history


def test_input_messages() -> None:
    runnable = RunnableLambda(
        lambda messages: "you said: "
        + "\n".join(str(m.content) for m in messages if isinstance(m, HumanMessage))
    )
    store: Dict = {}
    get_session_history = _get_get_session_history(store=store)
    with_history = RunnableWithMessageHistory(runnable, get_session_history)
    config: RunnableConfig = {"configurable": {"session_id": "1"}}
    output = with_history.invoke([HumanMessage(content="hello")], config)
    assert output == "you said: hello"
    output = with_history.invoke([HumanMessage(content="good bye")], config)
    assert output == "you said: hello\ngood bye"
    assert store == {
        "1": ChatMessageHistory(
            messages=[
                HumanMessage(content="hello"),
                AIMessage(content="you said: hello"),
                HumanMessage(content="good bye"),
                AIMessage(content="you said: hello\ngood bye"),
            ]
        )
    }


def test_input_dict() -> None:
    runnable = RunnableLambda(
        lambda input: "you said: "
        + "\n".join(
            str(m.content) for m in input["messages"] if isinstance(m, HumanMessage)
        )
    )
    get_session_history = _get_get_session_history()
    with_history = RunnableWithMessageHistory(
        runnable, get_session_history, input_messages_key="messages"
    )
    config: RunnableConfig = {"configurable": {"session_id": "2"}}
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
    with_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    config: RunnableConfig = {"configurable": {"session_id": "3"}}
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
    with_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    config: RunnableConfig = {"configurable": {"session_id": "4"}}
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
    with_history = RunnableWithMessageHistory(
        runnable,  # type: ignore
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    config: RunnableConfig = {"configurable": {"session_id": "5"}}
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
    with_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="output",
    )
    config: RunnableConfig = {"configurable": {"session_id": "6"}}
    output = with_history.invoke({"input": "hello"}, config)
    assert output == {"output": [AIMessage(content="you said: hello")]}
    output = with_history.invoke({"input": "good bye"}, config)
    assert output == {"output": [AIMessage(content="you said: hello\ngood bye")]}


def test_get_input_schema_input_dict() -> None:
    class RunnableWithChatHistoryInput(BaseModel):
        input: Union[str, BaseMessage, Sequence[BaseMessage]]

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
    with_history = RunnableWithMessageHistory(
        runnable,
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
    with_history = RunnableWithMessageHistory(
        runnable, get_session_history, output_messages_key="output"
    )
    assert (
        with_history.get_input_schema().schema()
        == RunnableWithChatHistoryInput.schema()
    )


def test_using_custom_config_specs() -> None:
    """Test that we can configure which keys should be passed to the session factory."""

    def _fake_llm(input: Dict[str, Any]) -> List[BaseMessage]:
        messages = input["messages"]
        return [
            AIMessage(
                content="you said: "
                + "\n".join(
                    str(m.content) for m in messages if isinstance(m, HumanMessage)
                )
            )
        ]

    runnable = RunnableLambda(_fake_llm)
    store = {}

    def get_session_history(user_id: str, conversation_id: str) -> ChatMessageHistory:
        if (user_id, conversation_id) not in store:
            store[(user_id, conversation_id)] = ChatMessageHistory()
        return store[(user_id, conversation_id)]

    with_message_history = RunnableWithMessageHistory(
        runnable,  # type: ignore
        get_session_history=get_session_history,
        input_messages_key="messages",
        history_messages_key="history",
        history_factory_config=[
            ConfigurableFieldSpec(
                id="user_id",
                annotation=str,
                name="User ID",
                description="Unique identifier for the user.",
                default="",
                is_shared=True,
            ),
            ConfigurableFieldSpec(
                id="conversation_id",
                annotation=str,
                name="Conversation ID",
                description="Unique identifier for the conversation.",
                default=None,
                is_shared=True,
            ),
        ],
    )
    result = with_message_history.invoke(
        {
            "messages": [HumanMessage(content="hello")],
        },
        {"configurable": {"user_id": "user1", "conversation_id": "1"}},
    )
    assert result == [
        AIMessage(content="you said: hello"),
    ]
    assert store == {
        ("user1", "1"): ChatMessageHistory(
            messages=[
                HumanMessage(content="hello"),
                AIMessage(content="you said: hello"),
            ]
        )
    }

    result = with_message_history.invoke(
        {
            "messages": [HumanMessage(content="goodbye")],
        },
        {"configurable": {"user_id": "user1", "conversation_id": "1"}},
    )
    assert result == [
        AIMessage(content="you said: goodbye"),
    ]
    assert store == {
        ("user1", "1"): ChatMessageHistory(
            messages=[
                HumanMessage(content="hello"),
                AIMessage(content="you said: hello"),
                HumanMessage(content="goodbye"),
                AIMessage(content="you said: goodbye"),
            ]
        )
    }

    result = with_message_history.invoke(
        {
            "messages": [HumanMessage(content="meow")],
        },
        {"configurable": {"user_id": "user2", "conversation_id": "1"}},
    )
    assert result == [
        AIMessage(content="you said: meow"),
    ]
    assert store == {
        ("user1", "1"): ChatMessageHistory(
            messages=[
                HumanMessage(content="hello"),
                AIMessage(content="you said: hello"),
                HumanMessage(content="goodbye"),
                AIMessage(content="you said: goodbye"),
            ]
        ),
        ("user2", "1"): ChatMessageHistory(
            messages=[
                HumanMessage(content="meow"),
                AIMessage(content="you said: meow"),
            ]
        ),
    }
