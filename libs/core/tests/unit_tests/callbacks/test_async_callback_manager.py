"""Unit tests for verifying event dispatching.

Much of this code is indirectly tested already through many end-to-end tests
that generate traces based on the callbacks. The traces are all verified
via snapshot testing (e.g., see unit tests for runnables).
"""

import contextvars
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

import pytest
from typing_extensions import override

from langchain_core.callbacks import (
    AsyncCallbackHandler,
    AsyncCallbackManager,
    BaseCallbackHandler,
)


async def test_inline_handlers_share_parent_context() -> None:
    """Verify that handlers that are configured to run_inline can update parent context.

    This test was created because some of the inline handlers were getting
    their own context as the handling logic was kicked off using asyncio.gather
    which does not automatically propagate the parent context (by design).

    This issue was affecting only a few specific handlers:

    * on_llm_start
    * on_chat_model_start

    which in some cases were triggered with multiple prompts and as a result
    triggering multiple tasks that were launched in parallel.
    """
    some_var: contextvars.ContextVar[str] = contextvars.ContextVar("some_var")

    class CustomHandler(AsyncCallbackHandler):
        """A handler that sets the context variable.

        The handler sets the context variable to the name of the callback that was
        called.
        """

        def __init__(self, *, run_inline: bool) -> None:
            """Initialize the handler."""
            self.run_inline = run_inline

        @override
        async def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
            """Update the callstack with the name of the callback."""
            some_var.set("on_llm_start")

    # The manager serves as a callback dispatcher.
    # It's responsible for dispatching callbacks to all registered handlers.
    manager = AsyncCallbackManager(handlers=[CustomHandler(run_inline=True)])

    # Check on_llm_start
    some_var.set("unset")
    await manager.on_llm_start({}, ["prompt 1"])
    assert some_var.get() == "on_llm_start"

    # Check what happens when run_inline is False
    # We don't expect the context to be updated
    manager2 = AsyncCallbackManager(
        handlers=[
            CustomHandler(run_inline=False),
        ]
    )

    some_var.set("unset")
    await manager2.on_llm_start({}, ["prompt 1"])
    # Will not be updated because the handler is not inline
    assert some_var.get() == "unset"


async def test_inline_handlers_share_parent_context_multiple() -> None:
    """A slightly more complex variation of the test unit test above.

    This unit test verifies that things work correctly when there are multiple prompts,
    and multiple handlers that are configured to run inline.
    """
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
        def __init__(self, name: str, *, run_inline: bool = True):
            self.name = name
            self.run_inline = run_inline

        async def on_llm_start(
            self,
            serialized: dict[str, Any],
            prompts: list[str],
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
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
                serialized,
                prompts,
                run_id=run_id,
                parent_run_id=parent_run_id,
                **kwargs,
            )

    handlers: list[BaseCallbackHandler] = [
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
        ]


async def test_shielded_callback_context_preservation() -> None:
    """Verify that shielded callbacks preserve context variables.

    This test specifically addresses the issue where async callbacks decorated
    with @shielded do not properly preserve context variables, breaking
    instrumentation and other context-dependent functionality.

    The issue manifests in callbacks that use the @shielded decorator:
    * on_llm_end
    * on_llm_error
    * on_chain_end
    * on_chain_error
    * And other shielded callback methods
    """
    context_var: contextvars.ContextVar[str] = contextvars.ContextVar("test_context")

    class ContextTestHandler(AsyncCallbackHandler):
        """Handler that reads context variables in shielded callbacks."""

        def __init__(self) -> None:
            self.run_inline = False
            self.context_values: list[str] = []

        @override
        async def on_llm_end(self, response: Any, **kwargs: Any) -> None:
            """This method is decorated with @shielded in the run manager."""
            # This should preserve the context variable value
            self.context_values.append(context_var.get("not_found"))

        @override
        async def on_chain_end(self, outputs: Any, **kwargs: Any) -> None:
            """This method is decorated with @shielded in the run manager."""
            # This should preserve the context variable value
            self.context_values.append(context_var.get("not_found"))

    # Set up the test context
    context_var.set("test_value")
    handler = ContextTestHandler()
    manager = AsyncCallbackManager(handlers=[handler])

    # Create run managers that have the shielded methods
    llm_managers = await manager.on_llm_start({}, ["test prompt"])
    llm_run_manager = llm_managers[0]

    chain_run_manager = await manager.on_chain_start({}, {"test": "input"})

    # Test LLM end callback (which is shielded)
    await llm_run_manager.on_llm_end({"response": "test"})  # type: ignore[arg-type]

    # Test Chain end callback (which is shielded)
    await chain_run_manager.on_chain_end({"output": "test"})

    # The context should be preserved in shielded callbacks
    # This was the main issue - shielded decorators were not preserving context
    assert handler.context_values == ["test_value", "test_value"], (
        f"Expected context values ['test_value', 'test_value'], "
        f"but got {handler.context_values}. "
        f"This indicates the shielded decorator is not preserving context variables."
    )


def test_configure_tracing_keyword_only() -> None:
    """Test that tracing parameter must be passed as keyword-only argument.

    This test verifies that the tracing parameter in AsyncCallbackManager.configure()
    is keyword-only and cannot be passed as a positional argument.
    """
    # This should work - tracing as keyword argument
    manager = AsyncCallbackManager.configure(tracing=True)
    assert isinstance(manager, AsyncCallbackManager)

    # This should work - tracing with default value (False)
    manager = AsyncCallbackManager.configure()
    assert isinstance(manager, AsyncCallbackManager)

    # This should work - all parameters as keyword arguments
    manager = AsyncCallbackManager.configure(
        inheritable_callbacks=None,
        local_callbacks=None,
        verbose=False,
        inheritable_tags=["tag1"],
        local_tags=["tag2"],
        inheritable_metadata={"key": "value"},
        local_metadata={"local_key": "local_value"},
        tracing=False,
    )
    assert isinstance(manager, AsyncCallbackManager)

    # This should fail - tracing as positional argument
    # Using noqa/type:ignore to suppress linting since we're testing incorrect usage
    with pytest.raises(
        TypeError, match="takes from 1 to 8 positional arguments but 9 were given"
    ):
        AsyncCallbackManager.configure(  # type: ignore[misc]
            None,  # inheritable_callbacks
            None,  # local_callbacks
            False,  # verbose  # noqa: FBT003
            None,  # inheritable_tags
            None,  # local_tags
            None,  # inheritable_metadata
            None,  # local_metadata
            False,  # tracing - should fail as positional  # noqa: FBT003
        )


def test_configure_with_tracing_enabled() -> None:
    """Test that configure works correctly with tracing enabled.

    This test verifies that the tracing parameter is properly handled
    when set to True.
    """
    # Configure with tracing enabled
    manager = AsyncCallbackManager.configure(tracing=True)
    assert isinstance(manager, AsyncCallbackManager)

    # Verify that handlers are set up (when tracing is enabled, a tracer
    # should be added). Note: The actual tracer setup depends on environment
    # variables like LANGCHAIN_TRACING_V2. So we just verify the manager is
    # created successfully.
    assert manager is not None


def test_configure_with_callbacks_and_tracing() -> None:
    """Test configure with both callbacks and tracing parameter.

    This test verifies that the configure method works correctly when
    both callbacks and the tracing parameter are provided.
    """
    handler = BaseCallbackHandler()

    manager = AsyncCallbackManager.configure(
        inheritable_callbacks=[handler],
        tracing=False,
    )

    assert isinstance(manager, AsyncCallbackManager)
    assert handler in manager.handlers
