import re
from collections.abc import Awaitable, Sequence
from typing import Any, Callable, Optional, Union

import pytest
from packaging import version
from pydantic import BaseModel
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
from langchain_core.utils.pydantic import PYDANTIC_VERSION
from tests.unit_tests.pydantic_utils import _schema


def test_interfaces() -> None:
    history = InMemoryChatMessageHistory()
    history.add_message(SystemMessage(content="system"))
    history.add_user_message("human 1")
    history.add_ai_message("ai")
    history.add_message(HumanMessage(content="human 2"))
    assert str(history) == "System: system\nHuman: human 1\nAI: ai\nHuman: human 2"


def _get_get_session_history(
    *,
    store: Optional[dict[str, Any]] = None,
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
    store: dict = {}
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
    store: dict = {}
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
    ] == ["you said: hello\ngood bye\nhi again"]
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
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
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
        input: Union[str, BaseMessage, Sequence[BaseMessage]]

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

    expected_schema: dict = {
        "title": "RunnableWithChatHistoryOutput",
        "type": "object",
    }
    if version.parse("2.11") <= PYDANTIC_VERSION:
        expected_schema["additionalProperties"] = True
    assert _schema(output_type) == expected_schema


def test_get_input_schema_input_messages() -> None:
    from pydantic import RootModel

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
    from langchain_core.tracers.root_listeners import AsyncListener

    def with_listeners(
        self,
        *,
        on_start: Optional[
            Union[Callable[[Run], None], Callable[[Run, RunnableConfig], None]]
        ] = None,
        on_end: Optional[
            Union[Callable[[Run], None], Callable[[Run, RunnableConfig], None]]
        ] = None,
        on_error: Optional[
            Union[Callable[[Run], None], Callable[[Run, RunnableConfig], None]]
        ] = None,
    ) -> Runnable[Input, Output]:
        from langchain_core.tracers.root_listeners import RootListenersTracer

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
        on_start: Optional[AsyncListener] = None,
        on_end: Optional[AsyncListener] = None,
        on_error: Optional[AsyncListener] = None,
    ) -> Runnable[Input, Output]:
        from langchain_core.tracers.root_listeners import AsyncRootListenersTracer

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
    store: dict = {}
    get_session_history = _get_get_session_history(store=store)
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
    store: dict = {}
    get_session_history = _get_get_session_history(store=store)
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


def _get_async_get_session_history(
    *,
    store: Optional[dict[str, Any]] = None,
) -> Callable[..., Awaitable[InMemoryChatMessageHistory]]:
    """Create an async version of get_session_history for testing."""
    chat_history_store = store if store is not None else {}

    async def get_session_history(
        session_id: str, **_kwargs: Any
    ) -> InMemoryChatMessageHistory:
        if session_id not in chat_history_store:
            chat_history_store[session_id] = InMemoryChatMessageHistory()
        return chat_history_store[session_id]

    return get_session_history


def test_async_get_session_history_sync_context() -> None:
    """Test that async get_session_history works when called from sync context."""
    runnable = RunnableLambda(
        lambda messages: "you said: "
        + "\n".join(str(m.content) for m in messages if isinstance(m, HumanMessage))
    )
    store: dict = {}
    get_session_history = _get_async_get_session_history(store=store)
    with_history = RunnableWithMessageHistory(runnable, get_session_history)
    config: RunnableConfig = {"configurable": {"session_id": "async_sync_1"}}

    # This test should raise an error when trying to use async session history
    # in a sync context where an event loop might be running
    try:
        output = with_history.invoke([HumanMessage(content="hello")], config)
        # If we get here without error, the async session history worked in sync context
        assert output == "you said: hello"

        # Second invocation should include history
        output = with_history.invoke([HumanMessage(content="good bye")], config)
        assert output == "you said: hello\ngood bye"

        # Verify the store contains the expected messages
        assert store == {
            "async_sync_1": InMemoryChatMessageHistory(
                messages=[
                    HumanMessage(content="hello"),
                    AIMessage(content="you said: hello"),
                    HumanMessage(content="good bye"),
                    AIMessage(content="you said: hello\ngood bye"),
                ]
            )
        }
    except RuntimeError as e:
        # This is expected when there's an event loop running
        if "running event loop" in str(e):
            pytest.skip(
                "Async session history not supported in sync context\
                    with running event loop"
            )
        else:
            raise


async def test_async_get_session_history_async_context() -> None:
    """Test that async get_session_history works when called from async context."""
    runnable = RunnableLambda(
        lambda messages: "you said: "
        + "\n".join(str(m.content) for m in messages if isinstance(m, HumanMessage))
    )
    store: dict = {}
    get_session_history = _get_get_session_history(store=store)
    with_history = RunnableWithMessageHistory(runnable, get_session_history)
    config: RunnableConfig = {"configurable": {"session_id": "async_async_1"}}

    # First invocation
    output = await with_history.ainvoke([HumanMessage(content="hello")], config)
    assert output == "you said: hello"

    # Second invocation should include history
    output = await with_history.ainvoke([HumanMessage(content="good bye")], config)
    assert output == "you said: hello\ngood bye"

    # Verify the store contains the expected messages
    assert store == {
        "async_async_1": InMemoryChatMessageHistory(
            messages=[
                HumanMessage(content="hello"),
                AIMessage(content="you said: hello"),
                HumanMessage(content="good bye"),
                AIMessage(content="you said: hello\ngood bye"),
            ]
        )
    }


def test_async_get_session_history_with_custom_config() -> None:
    """Test async get_session_history with custom configuration fields."""

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

    async def get_session_history(
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
            "messages": [HumanMessage(content="hello async")],
        },
        {"configurable": {"user_id": "async_user1", "conversation_id": "async_conv1"}},
    )
    assert result == [
        AIMessage(content="you said: hello async"),
    ]
    assert store == {
        ("async_user1", "async_conv1"): InMemoryChatMessageHistory(
            messages=[
                HumanMessage(content="hello async"),
                AIMessage(content="you said: hello async"),
            ]
        )
    }


async def test_async_get_session_history_with_custom_config_async() -> None:
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
            "messages": [HumanMessage(content="hello async")],
        },
        {"configurable": {"user_id": "async_user2", "conversation_id": "async_conv2"}},
    )
    assert result == [
        AIMessage(content="you said: hello async"),
    ]
    assert store == {
        ("async_user2", "async_conv2"): InMemoryChatMessageHistory(
            messages=[
                HumanMessage(content="hello async"),
                AIMessage(content="you said: hello async"),
            ]
        )
    }


def test_async_get_session_history_no_params() -> None:
    """Test async get_session_history with no parameters (using default history)."""
    runnable = RunnableLambda(
        lambda messages: "you said: "
        + "\n".join(str(m.content) for m in messages if isinstance(m, HumanMessage))
    )

    history = InMemoryChatMessageHistory()

    async def get_session_history() -> InMemoryChatMessageHistory:
        return history

    with_message_history = RunnableWithMessageHistory(runnable, get_session_history)

    # First call
    output = with_message_history.invoke([HumanMessage(content="hello")])
    assert output == "you said: hello"

    # Second call should include history
    output = with_message_history.invoke([HumanMessage(content="world")])
    assert output == "you said: hello\nworld"

    assert len(history.messages) == 4  # 2 human + 2 AI messages


async def test_async_get_session_history_no_params_async() -> None:
    """Test async get_session_history with no parameters in async context."""
    runnable = RunnableLambda(
        lambda messages: "you said: "
        + "\n".join(str(m.content) for m in messages if isinstance(m, HumanMessage))
    )

    history = InMemoryChatMessageHistory()

    def get_session_history() -> InMemoryChatMessageHistory:
        return history

    with_message_history = RunnableWithMessageHistory(runnable, get_session_history)

    # First call
    output = await with_message_history.ainvoke([HumanMessage(content="hello")])
    assert output == "you said: hello"

    # Second call should include history
    output = await with_message_history.ainvoke([HumanMessage(content="world")])
    assert output == "you said: hello\nworld"

    assert len(history.messages) == 4  # 2 human + 2 AI messages


def test_async_get_session_history_stream() -> None:
    """Test async get_session_history with streaming."""
    runnable = RunnableLambda(
        lambda messages: "you said: "
        + "\n".join(str(m.content) for m in messages if isinstance(m, HumanMessage))
    )
    store: dict = {}
    get_session_history = _get_async_get_session_history(store=store)
    with_history = RunnableWithMessageHistory(runnable, get_session_history)
    config: RunnableConfig = {"configurable": {"session_id": "stream_test"}}

    # First invocation
    output = list(with_history.stream([HumanMessage(content="hello")], config))
    assert output == ["you said: hello"]

    # Second invocation should include history
    output = list(with_history.stream([HumanMessage(content="world")], config))
    assert output == ["you said: hello\nworld"]


async def test_async_get_session_history_astream() -> None:
    """Test async get_session_history with async streaming."""
    runnable = RunnableLambda(
        lambda messages: "you said: "
        + "\n".join(str(m.content) for m in messages if isinstance(m, HumanMessage))
    )
    store: dict = {}
    get_session_history = _get_get_session_history(store=store)
    with_history = RunnableWithMessageHistory(runnable, get_session_history)
    config: RunnableConfig = {"configurable": {"session_id": "astream_test"}}

    # First invocation
    output = [
        chunk
        async for chunk in with_history.astream([HumanMessage(content="hello")], config)
    ]
    assert output == ["you said: hello"]

    # Second invocation should include history
    output = [
        chunk
        async for chunk in with_history.astream([HumanMessage(content="world")], config)
    ]
    assert output == ["you said: hello\nworld"]


def test_mixed_sync_async_session_history() -> None:
    """Test that we can use both sync and async session history functions."""
    # Test sync function first
    runnable = RunnableLambda(lambda messages: "sync: " + str(len(messages)))
    sync_store: dict = {}
    sync_get_session_history = _get_get_session_history(store=sync_store)
    sync_with_history = RunnableWithMessageHistory(runnable, sync_get_session_history)

    sync_output = sync_with_history.invoke(
        [HumanMessage(content="test")], {"configurable": {"session_id": "sync_test"}}
    )
    assert sync_output == "sync: 1"

    # Test async function
    async_store: dict = {}
    async_get_session_history = _get_async_get_session_history(store=async_store)
    async_with_history = RunnableWithMessageHistory(runnable, async_get_session_history)

    async_output = async_with_history.invoke(
        [HumanMessage(content="test")], {"configurable": {"session_id": "async_test"}}
    )
    assert async_output == "sync: 1"

    # Verify both stores are separate
    assert len(sync_store) == 1
    assert len(async_store) == 1
    assert "sync_test" in sync_store
    assert "async_test" in async_store


def test_async_get_session_history_error_handling() -> None:
    """Test error handling in async get_session_history functions."""
    runnable = RunnableLambda(lambda _: "test")

    async def failing_get_session_history(
        session_id: str,
    ) -> InMemoryChatMessageHistory:
        message = f"Failed to get history for {session_id}"
        raise ValueError(message)

    with_history = RunnableWithMessageHistory(runnable, failing_get_session_history)
    config: RunnableConfig = {"configurable": {"session_id": "test"}}

    # Should propagate the error from the async function
    with pytest.raises(ValueError, match="Failed to get history for test"):
        with_history.invoke([HumanMessage(content="hello")], config)


async def test_async_get_session_history_error_handling_async() -> None:
    """Test error handling in async get_session_history functions in async context."""
    runnable = RunnableLambda(lambda _: "test")

    def failing_get_session_history(
        session_id: str,
    ) -> InMemoryChatMessageHistory:
        message = f"Failed to get history for {session_id}"
        raise ValueError(message)

    with_history = RunnableWithMessageHistory(runnable, failing_get_session_history)
    config: RunnableConfig = {"configurable": {"session_id": "test"}}

    # Should propagate the error from the async function
    with pytest.raises(ValueError, match="Failed to get history for test"):
        await with_history.ainvoke([HumanMessage(content="hello")], config)


def test_config_handling_missing_configurable() -> None:
    """Test that missing 'configurable' key in config is handled properly."""
    runnable = RunnableLambda(lambda _: "test")

    history = InMemoryChatMessageHistory()

    async def get_session_history() -> InMemoryChatMessageHistory:
        return history

    with_message_history = RunnableWithMessageHistory(runnable, get_session_history)

    # Should work even with empty config (no 'configurable' key)
    output = with_message_history.invoke([HumanMessage(content="hello")], {})
    assert output == "test"

    # Should work with None config
    output = with_message_history.invoke([HumanMessage(content="hello")])
    assert output == "test"


def test_config_handling_empty_configurable() -> None:
    """Test that empty 'configurable' dict is handled properly."""
    runnable = RunnableLambda(lambda _: "test")

    history = InMemoryChatMessageHistory()

    async def get_session_history() -> InMemoryChatMessageHistory:
        return history

    with_message_history = RunnableWithMessageHistory(runnable, get_session_history)

    # Should work with empty configurable dict
    output = with_message_history.invoke(
        [HumanMessage(content="hello")], {"configurable": {}}
    )
    assert output == "test"


def test_async_session_history_with_dict_input_output() -> None:
    """Test async session history with dict input and output."""
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
    store: dict = {}
    get_session_history = _get_async_get_session_history(store=store)
    with_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="output",
    )
    config: RunnableConfig = {"configurable": {"session_id": "dict_test"}}

    output = with_history.invoke({"input": "hello"}, config)
    assert output == {"output": [AIMessage(content="you said: hello")]}

    output = with_history.invoke({"input": "world"}, config)
    assert output == {"output": [AIMessage(content="you said: hello\nworld")]}


async def test_async_session_history_with_dict_input_output_async() -> None:
    """Test async session history with dict input and output in async context."""
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
    store: dict = {}
    get_session_history = _get_get_session_history(store=store)
    with_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="output",
    )
    config: RunnableConfig = {"configurable": {"session_id": "dict_test_async"}}

    output = await with_history.ainvoke({"input": "hello"}, config)
    assert output == {"output": [AIMessage(content="you said: hello")]}

    output = await with_history.ainvoke({"input": "world"}, config)
    assert output == {"output": [AIMessage(content="you said: hello\nworld")]}


def test_async_session_history_with_chat_model() -> None:
    """Test async session history with a chat model."""
    store: dict = {}
    get_session_history = _get_async_get_session_history(store=store)

    runnable = LengthChatModel()
    with_history = RunnableWithMessageHistory(runnable, get_session_history)
    config: RunnableConfig = {"configurable": {"session_id": "chat_model_test"}}

    # First call - should return "1" (1 message)
    output = with_history.invoke([HumanMessage(content="hi")], config)
    assert output.content == "1"

    # Second call - should return "3" (previous human + AI + new human)
    output = with_history.invoke([HumanMessage(content="hi again")], config)
    assert output.content == "3"


async def test_async_session_history_with_chat_model_async() -> None:
    """Test async session history with a chat model in async context."""
    store: dict = {}
    get_session_history = _get_get_session_history(store=store)

    runnable = LengthChatModel()
    with_history = RunnableWithMessageHistory(runnable, get_session_history)
    config: RunnableConfig = {"configurable": {"session_id": "chat_model_test_async"}}

    # First call - should return "1" (1 message)
    output = await with_history.ainvoke([HumanMessage(content="hi")], config)
    assert output.content == "1"

    # Second call - should return "3" (previous human + AI + new human)
    output = await with_history.ainvoke([HumanMessage(content="hi again")], config)
    assert output.content == "3"


def test_run_get_session_history_method_directly() -> None:
    """Test the _run_get_session_history method directly with different scenarios."""
    store: dict = {}

    # Test with sync function
    sync_get_session_history = _get_get_session_history(store=store)
    runnable = RunnableLambda(lambda x: x)
    with_history = RunnableWithMessageHistory(runnable, sync_get_session_history)

    # Call _run_get_session_history directly
    history1 = with_history._run_get_session_history("test_session_1")
    assert isinstance(history1, InMemoryChatMessageHistory)
    assert "test_session_1" in store

    # Test with async function
    async_store: dict = {}
    async_get_session_history = _get_async_get_session_history(store=async_store)
    async_with_history = RunnableWithMessageHistory(runnable, async_get_session_history)

    # Call _run_get_session_history directly with async function
    history2 = async_with_history._run_get_session_history("test_session_2")
    assert isinstance(history2, InMemoryChatMessageHistory)
    assert "test_session_2" in async_store

    # Verify stores are separate
    assert "test_session_1" not in async_store
    assert "test_session_2" not in store
