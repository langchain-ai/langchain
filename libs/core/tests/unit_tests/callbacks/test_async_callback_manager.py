import contextvars
from contextlib import asynccontextmanager
from typing import Any, Optional
from uuid import UUID

import pytest

from langchain_core.callbacks import (
    AsyncCallbackHandler,
    AsyncCallbackManager,
)

counter_var = contextvars.ContextVar("counter", default=0)

shared_stack = []


@asynccontextmanager
async def set_counter_var() -> Any:
    token = counter_var.set(0)
    try:
        yield
    finally:
        counter_var.reset(token)


class StatefulAsyncCallbackHandler(AsyncCallbackHandler):
    def __init__(self, name: str, run_inline: bool = True):
        self.name = name
        self.run_inline = run_inline

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        if self.name == "StateModifier":
            current_counter = counter_var.get()
            counter_var.set(current_counter + 1)
            state = counter_var.get()
        elif self.name == "StateReader":
            state = counter_var.get()
        else:
            state = None

        shared_stack.append(state)

        await super().on_llm_start(
            serialized, prompts, run_id=run_id, parent_run_id=parent_run_id, **kwargs
        )


@pytest.mark.xfail(reason="Context is not maintained across async calls")
@pytest.mark.asyncio
async def test_async_callback_manager_context_loss() -> None:
    handlers = [
        StatefulAsyncCallbackHandler("StateModifier", run_inline=True),
        StatefulAsyncCallbackHandler("StateReader", run_inline=True),
        StatefulAsyncCallbackHandler("NonInlineHandler", run_inline=False),
    ]

    prompts = ["Prompt1", "Prompt2", "Prompt3"]

    async with set_counter_var():
        shared_stack.clear()
        manager = AsyncCallbackManager(handlers=handlers)
        await manager.on_llm_start({}, prompts)

        # Assert the order of states
        states = [entry for entry in shared_stack if entry is not None]
        assert states == [
            1,
            1,
            2,
            2,
            3,
            3,
        ], f"Expected order of states was broken due to context loss. Got {states}"
