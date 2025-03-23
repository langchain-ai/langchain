"""Unit tests for verifying event dispatching.

Much of this code is indirectly tested already through many end-to-end tests
that generate traces based on the callbacks. The traces are all verified
via snapshot testing (e.g., see unit tests for runnables).
"""

import contextvars
import json
import pickle
from contextlib import asynccontextmanager
from typing import Any, Optional
from uuid import UUID, uuid4

import pytest

from langchain_core.callbacks import (
    AsyncCallbackHandler,
    AsyncCallbackManager,
    AsyncCallbackManagerForToolRun,
    BaseCallbackHandler,
    BaseRunManager,
    CallbackManagerForToolRun,
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

        def __init__(self, run_inline: bool) -> None:
            """Initialize the handler."""
            self.run_inline = run_inline

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
        ], f"Expected order of states was broken due to context loss. Got {states}"


# Serialization Tests


async def test_base_run_manager_asdict() -> None:
    """Test that BaseRunManager._asdict() returns a properly serializable dictionary."""
    # Setup a manager with various types of metadata
    run_id = uuid4()
    parent_run_id = uuid4()
    tags = ["test", "serialization"]
    metadata = {"simple": "value", "number": 42, "complex_obj": object()}

    manager = BaseRunManager(
        run_id=run_id,
        parent_run_id=parent_run_id,
        tags=tags,
        metadata=metadata,
        handlers=[],  # Empty list of handlers
        inheritable_handlers=[],  # Empty list of inheritable handlers
    )

    # Get dictionary representation
    result = manager._asdict()

    # Verify all essential properties are included
    assert isinstance(result, dict)
    assert result["run_id"] == str(run_id)
    assert result["parent_run_id"] == str(parent_run_id)
    assert result["tags"] == tags
    assert "simple" in result["metadata"]
    assert result["metadata"]["simple"] == "value"
    assert result["metadata"]["number"] == 42
    # Complex objects should be filtered out
    assert "complex_obj" not in result["metadata"]

    # Should be JSON serializable
    serialized = json.dumps(result)
    assert isinstance(serialized, str)


async def test_callback_manager_json_serialization() -> None:
    """Test that AsyncCallbackManagerForToolRun can be properly JSON serialized."""
    # Setup callback manager
    run_id = uuid4()
    manager = AsyncCallbackManagerForToolRun(
        run_id=run_id,
        parent_run_id=None,
        tags=["tool_test"],
        metadata={"tool": "test_tool"},
        handlers=[],  # Empty list of handlers
        inheritable_handlers=[],  # Empty list of inheritable handlers
    )

    # Create tool arguments with the callback manager included
    tool_args = {
        "query": "test query",
        "run_manager": manager,
        "callbacks": manager.get_child(),
    }

    # Test JSON serialization
    try:
        serialized = json.dumps(
            tool_args,
            default=lambda obj: obj._asdict() if hasattr(obj, "_asdict") else str(obj),
        )
        # Successful serialization
        deserialized = json.loads(serialized)

        # Verify contents were preserved
        assert "run_manager" in deserialized
        assert deserialized["run_manager"]["run_id"] == str(run_id)
        assert "tool_test" in deserialized["run_manager"]["tags"]
        assert deserialized["run_manager"]["metadata"]["tool"] == "test_tool"
    except (TypeError, ValueError) as e:
        pytest.fail(f"JSON serialization failed: {e}")


def test_callback_manager_pickle_serialization() -> None:
    """Test callback managers can be
    pickled for msgpack serialization in checkpoints."""
    # Setup manager
    run_id = uuid4()
    manager = CallbackManagerForToolRun(
        run_id=run_id,
        tags=["pickle_test"],
        metadata={"serialization": "pickle"},
        handlers=[],  # Empty list of handlers
        inheritable_handlers=[],  # Empty list of inheritable handlers
    )

    # Test pickle serialization
    try:
        pickled = pickle.dumps(manager)
        unpickled = pickle.loads(pickled)

        # Verify properties survived
        assert str(unpickled.run_id) == str(run_id)
        assert "pickle_test" in unpickled.tags
        assert unpickled.metadata["serialization"] == "pickle"
    except (TypeError, ValueError) as e:
        pytest.fail(f"Pickle serialization failed: {e}")
