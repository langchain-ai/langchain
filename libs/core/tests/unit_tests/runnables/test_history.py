import re
from collections.abc import Callable, Sequence
from typing import Any

import pytest
from pydantic import BaseModel, RootModel
from typing_extensions import override

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.runnables.base import RunnableBinding, RunnableLambda
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.utils import ConfigurableFieldSpec, Input, Output
from langchain_core.tracers import Run
from langchain_core.tracers.root_listeners import (
    AsyncListener,
    AsyncRootListenersTracer,
    RootListenersTracer,
)
from tests.unit_tests.pydantic_utils import _schema


def test_interfaces() -> None:
    history = InMemoryChatMessageHistory()
    history.add_message(SystemMessage(content="system"))
    history.add_message(HumanMessage(content="human 1"))
    history.add_message(AIMessage(content="ai"))
    assert str(history) == "System: system\nHuman: human 1\nAI: ai"


def _get_get_session_history(
    *,
    store: dict[str, InMemoryChatMessageHistory] | None = None,
) -> Callable[..., InMemoryChatMessageHistory]:
    chat_history_store = store if store is not None else {}

    def get_session_history(
        session_id: str, **_kwargs: Any
    ) -> InMemoryChatMessageHistory:
        if session_id not in chat_history_store:
            chat_history_store[session_id] = InMemoryChatMessageHistory()
        return chat_history_store[session_id]

    return get_session_history


def test_input_messages() -> None:
    runnable = RunnableLambda(
        lambda messages: "you said: "
        + "\n".join(str(m.content) for m in messages if isinstance(m, HumanMessage))
    )
    store: dict[str, InMemoryChatMessageHistory] = {}
    get_session_history = _get_get_session_history(store=store)
    with_history = RunnableWithMessageHistory(runnable, get_session_history)
    config: RunnableConfig = {"configurable": {"session_id": "1"}}
    output = with_history.invoke([HumanMessage(content="hello")], config)
    assert output == "you said: hello"
    output = with_history.invoke([HumanMessage(content="good bye")], config)
    assert output == "you said: hello\ngood bye"
    output = [*with_history.stream([HumanMessage(content="hi again")], config)]
    assert output == ["you said: hello\ngood bye\nhi again"]
    assert store == {
        "1": InMemoryChatMessageHistory(
            messages=[
                HumanMessage(content="hello"),
                AIMessage(content="you said: hello"),
                HumanMessage(content="good bye"),
                AIMessage(content="you said: hello\ngood bye"),
                HumanMessage(content="hi again"),
                AIMessage(content="you said: hello\ngood bye\nhi again"),
            ]
        )
    }


async def test_input_messages_async() -> None:
    runnable = RunnableLambda(
        lambda messages: "you said: "
        + "\n".join(str(m.content) for m in messages if isinstance(m, HumanMessage))
    )
    store: dict[str, InMemoryChatMessageHistory] = {}
    get_session_history = _get_get_session_history(store=store)
    with_history = RunnableWithMessageHistory(runnable, get_session_history)
    config = {"session_id": "1_async"}
    output = await with_history.ainvoke([HumanMessage(content="hello")], config)  # type: ignore[arg-type]
    assert output == "you said: hello"
    output = await with_history.ainvoke([HumanMessage(content="good bye")], config)  # type: ignore[arg-type]
    assert output == "you said: hello\ngood bye"
    output = [
        c
        async for c in with_history.astream([HumanMessage(content="hi again")], config)  # type: ignore[arg-type]
    ]
    assert output == ["you said: hello\ngood bye\nhi again"]
    assert store == {
        "1_async": InMemoryChatMessageHistory(
            messages=[
                HumanMessage(content="hello"),
                AIMessage(content="you said: hello"),
                HumanMessage(content="good bye"),
                AIMessage(content="you said: hello\ngood bye"),
                HumanMessage(content="hi again"),
                AIMessage(content="you said: hello\ngood bye\nhi again"),
            ]
        )
    }


def test_input_dict() -> None:
    runnable = RunnableLambda(
        lambda params: "you said: "
        + "\n".join(
            str(m.content) for m in params["messages"] if isinstance(m, HumanMessage)
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


async def test_input_dict_async() -> None:
    runnable = RunnableLambda(
        lambda params: "you said: "
        + "\n".join(
            str(m.content) for m in params["messages"] if isinstance(m, HumanMessage)
        )
    )
    get_session_history = _get_get_session_history()
    with_history = RunnableWithMessageHistory(
        runnable, get_session_history, input_messages_key="messages"
    )
    config: RunnableConfig = {"configurable": {"session_id": "2_async"}}
    output = await with_history.ainvoke(
        {"messages": [HumanMessage(content="hello")]}, config
    )
    assert output == "you said: hello"
    output = await with_history.ainvoke(
        {"messages": [HumanMessage(content="good bye")]}, config
    )
    assert output == "you said: hello\ngood bye"


def test_input_dict_with_history_key() -> None:
    runnable = RunnableLambda(
        lambda params: "you said: "
        + "\n".join(
            [str(m.content) for m in params["history"] if isinstance(m, HumanMessage)]
            + [params["input"]]
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


async def test_input_dict_with_history_key_async() -> None:
    runnable = RunnableLambda(
        lambda params: "you said: "
        + "\n".join(
            [str(m.content) for m in params["history"] if isinstance(m, HumanMessage)]
            + [params["input"]]
        )
    )
    get_session_history = _get_get_session_history()
    with_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    config: RunnableConfig = {"configurable": {"session_id": "3_async"}}
    output = await with_history.ainvoke({"input": "hello"}, config)
    assert output == "you said: hello"
    output = await with_history.ainvoke({"input": "good bye"}, config)
    assert output == "you said: hello\ngood bye"


def test_output_message() -> None:
    runnable = RunnableLambda(
        lambda params: AIMessage(
            content="you said: "
            + "\n".join(
                [
                    str(m.content)
                    for m in params["history"]
                    if isinstance(m, HumanMessage)
                ]
                + [params["input"]]
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


async def test_output_message_async() -> None:
    runnable = RunnableLambda(
        lambda params: AIMessage(
            content="you said: "
            + "\n".join(
                [
                    str(m.content)
                    for m in params["history"]
                    if isinstance(m, HumanMessage)
                ]
                + [params["input"]]
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
    config: RunnableConfig = {"configurable": {"session_id": "4_async"}}
    output = await with_history.ainvoke({"input": "hello"}, config)
    assert output == AIMessage(content="you said: hello")
    output = await with_history.ainvoke({"input": "good bye"}, config)
    assert output == AIMessage(content="you said: hello\ngood bye")


class LengthChatModel(BaseChatModel):
    """A fake chat model that returns the length of the messages passed in."""

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call."""
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=str(len(messages))))]
        )

    @property
    def _llm_type(self) -> str:
        return "length-fake-chat-model"


def test_input_messages_output_message() -> None:
    runnable = LengthChatModel()
    get_session_history = _get_get_session_history()
    with_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
    )
    config: RunnableConfig = {"configurable": {"session_id": "5"}}
    output = with_history.invoke([HumanMessage(content="hi")], config)
    assert output.content == "1"
    output = with_history.invoke([HumanMessage(content="hi")], config)
    assert output.content == "3"


async def test_input_messages_output_message_async() -> None:
    runnable = LengthChatModel()
    get_session_history = _get_get_session_history()
    with_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
    )
    config: RunnableConfig = {"configurable": {"session_id": "5_async"}}
    output = await with_history.ainvoke([HumanMessage(content="hi")], config)
    assert output.content == "1"
    output = await with_history.ainvoke([HumanMessage(content="hi")], config)
    assert output.content == "3"


def test_output_messages() -> None:
    runnable = RunnableLambda(
        lambda params: [
            AIMessage(
                content="you said: "
                + "\n".join(
                    [
                        str(m.content)
                        for m in params["history"]
                        if isinstance(m, HumanMessage)
                    ]
                    + [params["input"]]
                )
            )
        ]
    )
    get_session_history = _get_get_session_history()
    with_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    config: RunnableConfig = {"configurable": {"session_id": "6"}}
    output = with_history.invoke({"input": "hello"}, config)
    assert output == [AIMessage(content="you said: hello")]
    output = with_history.invoke({"input": "good bye"}, config)
    assert output == [AIMessage(content="you said: hello\ngood bye")]


async def test_output_messages_async() -> None:
    runnable = RunnableLambda(
        lambda params: [
            AIMessage(
                content="you said: "
                + "\n".join(
                    [
                        str(m.content)
                        for m in params["history"]
                        if isinstance(m, HumanMessage)
                    ]
                    + [params["input"]]
                )
            )
        ]
    )
    get_session_history = _get_get_session_history()
    with_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    config: RunnableConfig = {"configurable": {"session_id": "6_async"}}
    output = await with_history.ainvoke({"input": "hello"}, config)
    assert output == [AIMessage(content="you said: hello")]
    output = await with_history.ainvoke({"input": "good bye"}, config)
    assert output == [AIMessage(content="you said: hello\ngood bye")]


def test_output_dict() -> None:
    runnable = RunnableLambda(
        lambda params: {
            "output": [
                AIMessage(
                    content="you said: "
                    + "\n".join(
                        [
                            str(m.content)
                            for m in params["history"]
                            if isinstance(m, HumanMessage)
                        ]
                        + [params["input"]]
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
    config: RunnableConfig = {"configurable": {"session_id": "7"}}
    output = with_history.invoke({"input": "hello"}, config)
    assert output == {"output": [AIMessage(content="you said: hello")]}
    output = with_history.invoke({"input": "good bye"}, config)
    assert output == {"output": [AIMessage(content="you said: hello\ngood bye")]}


async def test_output_dict_async() -> None:
    runnable = RunnableLambda(
        lambda params: {
            "output": [
                AIMessage(
                    content="you said: "
                    + "\n".join(
                        [
                            str(m.content)
                            for m in params["history"]
                            if isinstance(m, HumanMessage)
                        ]
                        + [params["input"]]
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
    config: RunnableConfig = {"configurable": {"session_id": "7_async"}}
    output = await with_history.ainvoke({"input": "hello"}, config)
    assert output == {"output": [AIMessage(content="you said: hello")]}
    output = await with_history.ainvoke({"input": "good bye"}, config)
    assert output == {"output": [AIMessage(content="you said: hello\ngood bye")]}


def test_get_input_schema_input_dict() -> None:
    class RunnableWithChatHistoryInput(BaseModel):
        input: str | BaseMessage | Sequence[BaseMessage]

    runnable = RunnableLambda(
        lambda params: {
            "output": [
                AIMessage(
                    content="you said: "
                    + "\n".join(
                        [
                            str(m.content)
                            for m in params["history"]
                            if isinstance(m, HumanMessage)
                        ]
                        + [params["input"]]
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
    assert _schema(with_history.get_input_schema()) == _schema(
        RunnableWithChatHistoryInput
    )


def test_get_output_schema() -> None:
    """Test get output schema."""
    runnable = RunnableLambda(
        lambda params: {
            "output": [
                AIMessage(
                    content="you said: "
                    + "\n".join(
                        [
                            str(m.content)
                            for m in params["history"]
                            if isinstance(m, HumanMessage)
                        ]
                        + [params["input"]]
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
    output_type = with_history.get_output_schema()

    expected_schema: dict[str, Any] = {
        "title": "RunnableWithChatHistoryOutput",
        "type": "object",
    }
    assert _schema(output_type) == expected_schema


def test_get_input_schema_input_messages() -> None:
    runnable_with_message_history_input = RootModel[Sequence[BaseMessage]]

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
    expected_schema = _schema(runnable_with_message_history_input)
    expected_schema["title"] = "RunnableWithChatHistoryInput"
    assert _schema(with_history.get_input_schema()) == expected_schema


def test_using_custom_config_specs() -> None:
    """Test that we can configure which keys should be passed to the session factory."""

    def _fake_llm(params: dict[str, Any]) -> list[BaseMessage]:
        messages = params["messages"]
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

    def get_session_history(
        user_id: str, conversation_id: str
    ) -> InMemoryChatMessageHistory:
        if (user_id, conversation_id) not in store:
            store[user_id, conversation_id] = InMemoryChatMessageHistory()
        return store[user_id, conversation_id]

    with_message_history = RunnableWithMessageHistory(
        runnable,
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
        ("user1", "1"): InMemoryChatMessageHistory(
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
        ("user1", "1"): InMemoryChatMessageHistory(
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
        ("user1", "1"): InMemoryChatMessageHistory(
            messages=[
                HumanMessage(content="hello"),
                AIMessage(content="you said: hello"),
                HumanMessage(content="goodbye"),
                AIMessage(content="you said: goodbye"),
            ]
        ),
        ("user2", "1"): InMemoryChatMessageHistory(
            messages=[
                HumanMessage(content="meow"),
                AIMessage(content="you said: meow"),
            ]
        ),
    }


async def test_using_custom_config_specs_async() -> None:
    """Test that we can configure which keys should be passed to the session factory."""

    def _fake_llm(params: dict[str, Any]) -> list[BaseMessage]:
        messages = params["messages"]
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

    def get_session_history(
        user_id: str, conversation_id: str
    ) -> InMemoryChatMessageHistory:
        if (user_id, conversation_id) not in store:
            store[user_id, conversation_id] = InMemoryChatMessageHistory()
        return store[user_id, conversation_id]

    with_message_history = RunnableWithMessageHistory(
        runnable,
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
    result = await with_message_history.ainvoke(
        {
            "messages": [HumanMessage(content="hello")],
        },
        {"configurable": {"user_id": "user1_async", "conversation_id": "1_async"}},
    )
    assert result == [
        AIMessage(content="you said: hello"),
    ]
    assert store == {
        ("user1_async", "1_async"): InMemoryChatMessageHistory(
            messages=[
                HumanMessage(content="hello"),
                AIMessage(content="you said: hello"),
            ]
        )
    }

    result = await with_message_history.ainvoke(
        {
            "messages": [HumanMessage(content="goodbye")],
        },
        {"configurable": {"user_id": "user1_async", "conversation_id": "1_async"}},
    )
    assert result == [
        AIMessage(content="you said: goodbye"),
    ]
    assert store == {
        ("user1_async", "1_async"): InMemoryChatMessageHistory(
            messages=[
                HumanMessage(content="hello"),
                AIMessage(content="you said: hello"),
                HumanMessage(content="goodbye"),
                AIMessage(content="you said: goodbye"),
            ]
        )
    }

    result = await with_message_history.ainvoke(
        {
            "messages": [HumanMessage(content="meow")],
        },
        {"configurable": {"user_id": "user2_async", "conversation_id": "1_async"}},
    )
    assert result == [
        AIMessage(content="you said: meow"),
    ]
    assert store == {
        ("user1_async", "1_async"): InMemoryChatMessageHistory(
            messages=[
                HumanMessage(content="hello"),
                AIMessage(content="you said: hello"),
                HumanMessage(content="goodbye"),
                AIMessage(content="you said: goodbye"),
            ]
        ),
        ("user2_async", "1_async"): InMemoryChatMessageHistory(
            messages=[
                HumanMessage(content="meow"),
                AIMessage(content="you said: meow"),
            ]
        ),
    }


def test_ignore_session_id() -> None:
    """Test without config."""

    def _fake_llm(messages: list[BaseMessage]) -> list[BaseMessage]:
        return [
            AIMessage(
                content="you said: "
                + "\n".join(
                    str(m.content) for m in messages if isinstance(m, HumanMessage)
                )
            )
        ]

    runnable = RunnableLambda(_fake_llm)
    history = InMemoryChatMessageHistory()
    with_message_history = RunnableWithMessageHistory(runnable, lambda: history)
    _ = with_message_history.invoke("hello")
    _ = with_message_history.invoke("hello again")
    assert len(history.messages) == 4


class _RunnableLambdaWithRaiseError(RunnableLambda[Input, Output]):
    def with_listeners(
        self,
        *,
        on_start: Callable[[Run], None]
        | Callable[[Run, RunnableConfig], None]
        | None = None,
        on_end: Callable[[Run], None]
        | Callable[[Run, RunnableConfig], None]
        | None = None,
        on_error: Callable[[Run], None]
        | Callable[[Run, RunnableConfig], None]
        | None = None,
    ) -> Runnable[Input, Output]:
        def create_tracer(config: RunnableConfig) -> RunnableConfig:
            tracer = RootListenersTracer(
                config=config,
                on_start=on_start,
                on_end=on_end,
                on_error=on_error,
            )
            tracer.raise_error = True
            return {
                "callbacks": [tracer],
            }

        return RunnableBinding(
            bound=self,
            config_factories=[create_tracer],
        )

    def with_alisteners(
        self,
        *,
        on_start: AsyncListener | None = None,
        on_end: AsyncListener | None = None,
        on_error: AsyncListener | None = None,
    ) -> Runnable[Input, Output]:
        def create_tracer(config: RunnableConfig) -> RunnableConfig:
            tracer = AsyncRootListenersTracer(
                config=config,
                on_start=on_start,
                on_end=on_end,
                on_error=on_error,
            )
            tracer.raise_error = True
            return {
                "callbacks": [tracer],
            }

        return RunnableBinding(
            bound=self,
            config_factories=[create_tracer],
        )


def test_get_output_messages_no_value_error() -> None:
    runnable = _RunnableLambdaWithRaiseError(
        lambda messages: "you said: "
        + "\n".join(str(m.content) for m in messages if isinstance(m, HumanMessage))
    )
    get_session_history = _get_get_session_history()
    with_history = RunnableWithMessageHistory(runnable, get_session_history)
    config: RunnableConfig = {
        "configurable": {"session_id": "1", "message_history": get_session_history("1")}
    }
    may_catch_value_error = None
    try:
        with_history.bound.invoke([HumanMessage(content="hello")], config)
    except ValueError as e:
        may_catch_value_error = e
    assert may_catch_value_error is None


def test_get_output_messages_with_value_error() -> None:
    illegal_bool_message = False
    runnable = _RunnableLambdaWithRaiseError(lambda _: illegal_bool_message)
    get_session_history = _get_get_session_history()
    with_history = RunnableWithMessageHistory(runnable, get_session_history)  # type: ignore[arg-type]
    config: RunnableConfig = {
        "configurable": {"session_id": "1", "message_history": get_session_history("1")}
    }

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Expected str, BaseMessage, list[BaseMessage], or tuple[BaseMessage]."
            f" Got {illegal_bool_message}."
        ),
    ):
        with_history.bound.invoke([HumanMessage(content="hello")], config)

    illegal_int_message = 123
    runnable2 = _RunnableLambdaWithRaiseError(lambda _: illegal_int_message)
    with_history = RunnableWithMessageHistory(runnable2, get_session_history)  # type: ignore[arg-type]

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Expected str, BaseMessage, list[BaseMessage], or tuple[BaseMessage]."
            f" Got {illegal_int_message}."
        ),
    ):
        with_history.bound.invoke([HumanMessage(content="hello")], config)
