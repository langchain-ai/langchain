"""Tests for agent model invocation config propagation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import Field
from typing_extensions import override

from langchain.agents import create_agent
from tests.unit_tests.agents.model import FakeToolCallingModel

if TYPE_CHECKING:
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.runnables import RunnableConfig


class _ConfigCapturingModel(FakeToolCallingModel):
    captured_configs: list[Any] = Field(default_factory=list, exclude=True)

    @override
    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> BaseMessage:
        self.captured_configs.append(config)
        return super().invoke(input, config=config, **kwargs)

    @override
    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> BaseMessage:
        self.captured_configs.append(config)
        return await super().ainvoke(input, config=config, **kwargs)


class _CallbackRecorder(BaseCallbackHandler):
    starts: list[dict[str, Any]]

    def __init__(self) -> None:
        self.starts = []

    @override
    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.starts.append({"metadata": metadata or {}, "messages": messages, "tags": tags or []})


class _AsyncCallbackRecorder(AsyncCallbackHandler):
    starts: list[dict[str, Any]]

    def __init__(self) -> None:
        self.starts = []

    @override
    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.starts.append({"metadata": metadata or {}, "messages": messages, "tags": tags or []})


def test_agent_model_receives_config_and_callbacks() -> None:
    """The agent model node should forward invocation config to the chat model."""
    model = _ConfigCapturingModel()
    handler = _CallbackRecorder()
    agent = create_agent(model=model)

    agent.invoke(
        {"messages": [HumanMessage("Hello")]},
        config={
            "callbacks": [handler],
            "configurable": {"thread_id": "sync-thread"},
            "metadata": {"test_case": "sync"},
            "tags": ["agent-config"],
        },
    )

    assert len(model.captured_configs) == 1
    captured_config = model.captured_configs[0]
    assert captured_config is not None
    assert captured_config["configurable"]["thread_id"] == "sync-thread"

    assert len(handler.starts) == 1
    assert handler.starts[0]["metadata"]["test_case"] == "sync"
    assert "agent-config" in handler.starts[0]["tags"]


async def test_agent_model_receives_config_and_callbacks_async() -> None:
    """The async agent model node should forward invocation config to the chat model."""
    model = _ConfigCapturingModel()
    handler = _AsyncCallbackRecorder()
    agent = create_agent(model=model)

    await agent.ainvoke(
        {"messages": [HumanMessage("Hello")]},
        config={
            "callbacks": [handler],
            "configurable": {"thread_id": "async-thread"},
            "metadata": {"test_case": "async"},
            "tags": ["agent-config-async"],
        },
    )

    assert len(model.captured_configs) == 1
    captured_config = model.captured_configs[0]
    assert captured_config is not None
    assert captured_config["configurable"]["thread_id"] == "async-thread"

    assert len(handler.starts) == 1
    assert handler.starts[0]["metadata"]["test_case"] == "async"
    assert "agent-config-async" in handler.starts[0]["tags"]
