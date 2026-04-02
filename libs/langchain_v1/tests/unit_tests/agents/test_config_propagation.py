"""Tests that RunnableConfig (including callbacks) propagates to model invocations."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage

from langchain.agents import create_agent
from tests.unit_tests.agents.model import FakeToolCallingModel


class _CapturingCallbackHandler(BaseCallbackHandler):
    """Records every on_llm_start call to verify callback propagation."""

    def __init__(self) -> None:
        self.llm_start_count = 0

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        self.llm_start_count += 1


def test_callbacks_propagate_to_model_invoke() -> None:
    """Config callbacks must reach model_.invoke inside the agent model node.

    Before the fix, _execute_model_sync called model_.invoke(messages) without
    forwarding the LangGraph config, so callbacks passed via agent.invoke(config=...)
    were silently dropped and never reached the underlying model call.
    """
    handler = _CapturingCallbackHandler()
    agent = create_agent(model=FakeToolCallingModel(tool_calls=[[]]))

    agent.invoke(
        {"messages": [HumanMessage("hello")]},
        config={"callbacks": [handler]},
    )

    assert handler.llm_start_count == 1, (
        f"Expected 1 on_llm_start call, got {handler.llm_start_count}. "
        "Config callbacks are not reaching model_.invoke."
    )


async def test_callbacks_propagate_to_model_ainvoke() -> None:
    """Config callbacks must reach model_.ainvoke inside the async agent model node.

    Before the fix, _execute_model_async called model_.ainvoke(messages) without
    forwarding the LangGraph config.
    """
    handler = _CapturingCallbackHandler()
    agent = create_agent(model=FakeToolCallingModel(tool_calls=[[]]))

    await agent.ainvoke(
        {"messages": [HumanMessage("hello")]},
        config={"callbacks": [handler]},
    )

    assert handler.llm_start_count == 1, (
        f"Expected 1 on_llm_start call, got {handler.llm_start_count}. "
        "Config callbacks are not reaching model_.ainvoke."
    )
