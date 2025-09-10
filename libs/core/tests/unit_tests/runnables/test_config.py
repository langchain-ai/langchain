import json
import uuid
from contextvars import Context, copy_context
from typing import Any, cast

import pytest

from langchain_core.callbacks.manager import (
    AsyncCallbackManager,
    CallbackManager,
    atrace_as_chain_group,
    trace_as_chain_group,
)
from langchain_core.callbacks.stdout import StdOutCallbackHandler
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.runnables import RunnableBinding, RunnablePassthrough
from langchain_core.runnables.config import (
    RunnableConfig,
    _set_config_context,
    ensure_config,
    merge_configs,
    run_in_executor,
    set_config_context,
)
from langchain_core.tracers.stdout import ConsoleCallbackHandler


def test_ensure_config() -> None:
    run_id = str(uuid.uuid4())
    arg: dict = {
        "something": "else",
        "metadata": {"foo": "bar"},
        "configurable": {"baz": "qux"},
        "callbacks": [StdOutCallbackHandler()],
        "tags": ["tag1", "tag2"],
        "max_concurrency": 1,
        "recursion_limit": 100,
        "run_id": run_id,
        "run_name": "test",
    }
    arg_str = json.dumps({**arg, "callbacks": []})
    ctx = copy_context()
    ctx.run(
        _set_config_context,
        {
            "callbacks": [ConsoleCallbackHandler()],
            "metadata": {"a": "b"},
            "configurable": {"c": "d"},
            "tags": ["tag3", "tag4"],
        },
    )
    config = ctx.run(ensure_config, cast("RunnableConfig", arg))
    assert len(arg["callbacks"]) == 1, (
        "ensure_config should not modify the original config"
    )
    assert json.dumps({**arg, "callbacks": []}) == arg_str, (
        "ensure_config should not modify the original config"
    )
    assert config is not arg
    assert config["callbacks"] is not arg["callbacks"]
    assert config["metadata"] is not arg["metadata"]
    assert config["configurable"] is not arg["configurable"]
    assert config == {
        "tags": ["tag1", "tag2"],
        "metadata": {"foo": "bar", "baz": "qux", "something": "else"},
        "callbacks": [arg["callbacks"][0]],
        "recursion_limit": 100,
        "configurable": {"baz": "qux", "something": "else"},
        "max_concurrency": 1,
        "run_id": run_id,
        "run_name": "test",
    }


async def test_merge_config_callbacks() -> None:
    manager: RunnableConfig = {
        "callbacks": CallbackManager(handlers=[StdOutCallbackHandler()])
    }
    handlers: RunnableConfig = {"callbacks": [ConsoleCallbackHandler()]}
    other_handlers: RunnableConfig = {"callbacks": [StreamingStdOutCallbackHandler()]}

    merged = merge_configs(manager, handlers)["callbacks"]

    assert isinstance(merged, CallbackManager)
    assert len(merged.handlers) == 2
    assert isinstance(merged.handlers[0], StdOutCallbackHandler)
    assert isinstance(merged.handlers[1], ConsoleCallbackHandler)

    merged = merge_configs(handlers, manager)["callbacks"]

    assert isinstance(merged, CallbackManager)
    assert len(merged.handlers) == 2
    assert isinstance(merged.handlers[0], StdOutCallbackHandler)
    assert isinstance(merged.handlers[1], ConsoleCallbackHandler)

    merged = merge_configs(handlers, other_handlers)["callbacks"]

    assert isinstance(merged, list)
    assert len(merged) == 2
    assert isinstance(merged[0], ConsoleCallbackHandler)
    assert isinstance(merged[1], StreamingStdOutCallbackHandler)

    # Check that the original object wasn't mutated
    merged = merge_configs(manager, handlers)["callbacks"]

    assert isinstance(merged, CallbackManager)
    assert len(merged.handlers) == 2
    assert isinstance(merged.handlers[0], StdOutCallbackHandler)
    assert isinstance(merged.handlers[1], ConsoleCallbackHandler)

    with trace_as_chain_group("test") as gm:
        group_manager: RunnableConfig = {
            "callbacks": gm,
        }
        merged = merge_configs(group_manager, handlers)["callbacks"]
        assert isinstance(merged, CallbackManager)
        assert len(merged.handlers) == 1
        assert isinstance(merged.handlers[0], ConsoleCallbackHandler)

        merged = merge_configs(handlers, group_manager)["callbacks"]
        assert isinstance(merged, CallbackManager)
        assert len(merged.handlers) == 1
        assert isinstance(merged.handlers[0], ConsoleCallbackHandler)
        merged = merge_configs(group_manager, manager)["callbacks"]
        assert isinstance(merged, CallbackManager)
        assert len(merged.handlers) == 1
        assert isinstance(merged.handlers[0], StdOutCallbackHandler)

    async with atrace_as_chain_group("test_async") as gm:
        group_manager = {
            "callbacks": gm,
        }
        merged = merge_configs(group_manager, handlers)["callbacks"]
        assert isinstance(merged, AsyncCallbackManager)
        assert len(merged.handlers) == 1
        assert isinstance(merged.handlers[0], ConsoleCallbackHandler)

        merged = merge_configs(handlers, group_manager)["callbacks"]
        assert isinstance(merged, AsyncCallbackManager)
        assert len(merged.handlers) == 1
        assert isinstance(merged.handlers[0], ConsoleCallbackHandler)
        merged = merge_configs(group_manager, manager)["callbacks"]
        assert isinstance(merged, AsyncCallbackManager)
        assert len(merged.handlers) == 1
        assert isinstance(merged.handlers[0], StdOutCallbackHandler)


def test_config_arbitrary_keys() -> None:
    base: RunnablePassthrough[Any] = RunnablePassthrough()
    bound = base.with_config(my_custom_key="my custom value")
    config = cast("RunnableBinding", bound).config

    assert config.get("my_custom_key") == "my custom value"


async def test_run_in_executor() -> None:
    def raises_stop_iter() -> Any:
        return next(iter([]))

    with pytest.raises(StopIteration):
        raises_stop_iter()

    with pytest.raises(RuntimeError):
        await run_in_executor(None, raises_stop_iter)


def test_set_config_context_reuse_raises_error() -> None:
    """Test that reusing the same context manager raises RuntimeError."""
    config = RunnableConfig(tags=["test"])
    ctx_manager = set_config_context(config)

    # First enter should work
    with ctx_manager as ctx1:
        assert ctx1 is not None

        # Second enter should raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            with ctx_manager:
                pass  # Should not reach here

        assert "Cannot re-enter an already-entered context manager" in str(
            exc_info.value
        )


def test_set_config_context_exit_without_enter() -> None:
    """Test that exiting without entering raises RuntimeError."""
    config = RunnableConfig(tags=["test"])
    ctx_manager = set_config_context(config)

    # Attempting to exit without entering should raise RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        ctx_manager.__exit__(None, None, None)

    assert "Cannot exit context manager that was not entered" in str(exc_info.value)


def test_set_config_context_normal_usage() -> None:
    """Test that normal usage still works correctly."""
    config = RunnableConfig(
        tags=["test"], metadata={"key": "value"}, configurable={"param": "value"}
    )

    # Normal usage should work without issues
    with set_config_context(config) as ctx:
        assert ctx is not None
        # Context should be a contextvars.Context object
        assert isinstance(ctx, Context)

    # After exiting, we should be able to create a new context manager
    # with the same config
    with set_config_context(config) as ctx2:
        assert ctx2 is not None
        assert isinstance(ctx2, Context)


def test_set_config_context_nested_different_instances() -> None:
    """Test that using different instances in nested contexts works properly."""
    from langchain_core.runnables.config import set_config_context

    config1 = RunnableConfig(tags=["outer"], metadata={"level": "outer"})
    config2 = RunnableConfig(tags=["inner"], metadata={"level": "inner"})

    # Nested contexts with different instances should work
    with set_config_context(config1) as ctx1:
        assert ctx1 is not None

        with set_config_context(config2) as ctx2:
            assert ctx2 is not None
            # Both contexts should be valid
            from contextvars import Context

            assert isinstance(ctx1, Context)
            assert isinstance(ctx2, Context)
            # They should be different contexts
            assert ctx1 is not ctx2

        # After inner context exits, outer should still be valid
        assert ctx1 is not None




