"""Callback emission helpers for the LangGraph-based agent."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from langchain_core.agents import AgentFinish
from langchain_core.callbacks.manager import (
    ahandle_event,
    handle_event,
)
from langchain_core.messages import AIMessage
from langchain_core.runnables.config import (
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
)

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

    from langchain.agents.middleware.types import AgentState


def _build_agent_finish(state: AgentState[Any]) -> AgentFinish:
    """Construct an `AgentFinish` from the current agent state.

    Extracts the final AI message content from the state's messages list.
    If no AI message is found, uses an empty string.

    Args:
        state: The current agent state.

    Returns:
        An `AgentFinish` with the final response content.
    """
    messages = state.get("messages", [])
    output = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            output = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    return AgentFinish(
        return_values={"output": output},
        log=output,
    )


def _emit_agent_finish(state: AgentState[Any], config: RunnableConfig) -> None:
    """Emit `on_agent_finish` to all registered callback handlers (sync).

    Args:
        state: The current agent state.
        config: The runnable config containing callback handlers.
    """
    callback_manager = get_callback_manager_for_config(config)
    if not callback_manager.handlers:
        return

    finish = _build_agent_finish(state)
    handle_event(
        callback_manager.handlers,
        "on_agent_finish",
        "ignore_agent",
        finish,
        run_id=uuid.uuid4(),
        parent_run_id=None,
        tags=callback_manager.tags,
    )


async def _aemit_agent_finish(state: AgentState[Any], config: RunnableConfig) -> None:
    """Emit `on_agent_finish` to all registered callback handlers (async).

    Args:
        state: The current agent state.
        config: The runnable config containing callback handlers.
    """
    callback_manager = get_async_callback_manager_for_config(config)
    if not callback_manager.handlers:
        return

    finish = _build_agent_finish(state)
    await ahandle_event(
        callback_manager.handlers,
        "on_agent_finish",
        "ignore_agent",
        finish,
        run_id=uuid.uuid4(),
        parent_run_id=None,
        tags=callback_manager.tags,
    )
