import json
import uuid
from contextvars import copy_context
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


def test_inherit_run_name_default_behavior() -> None:
    """Test that by default, run_name is NOT inherited to child runs."""
    from langchain_core.callbacks.manager import CallbackManager
    from langchain_core.runnables.config import patch_config

    # Create a real callback manager with a handler
    handler = StdOutCallbackHandler()
    callback_manager = CallbackManager(handlers=[handler])

    # Test default behavior (inherit_run_name not set or False)
    config: RunnableConfig = {
        "run_name": "parent_run",
        "callbacks": callback_manager,
    }

    # When callbacks are replaced, run_name should be deleted by default
    new_callback_manager = CallbackManager(handlers=[handler])
    patched = patch_config(config, callbacks=new_callback_manager)
    assert "run_name" not in patched, "run_name should be deleted by default"

    # Explicitly set inherit_run_name to False
    config_explicit_false: RunnableConfig = {
        "run_name": "parent_run",
        "inherit_run_name": False,
        "callbacks": callback_manager,
    }

    patched_explicit = patch_config(
        config_explicit_false, callbacks=new_callback_manager
    )
    assert "run_name" not in patched_explicit, (
        "run_name should be deleted when inherit_run_name=False"
    )


def test_inherit_run_name_enabled() -> None:
    """Test that when inherit_run_name=True, run_name is preserved for child runs."""
    from langchain_core.callbacks.manager import CallbackManager
    from langchain_core.runnables.config import patch_config

    # Create a real callback manager with a handler
    handler = StdOutCallbackHandler()
    callback_manager = CallbackManager(handlers=[handler])

    # Test with inherit_run_name=True
    config: RunnableConfig = {
        "run_name": "parent_run",
        "inherit_run_name": True,
        "callbacks": callback_manager,
    }

    # When callbacks are replaced, run_name should be preserved
    new_callback_manager = CallbackManager(handlers=[handler])
    patched = patch_config(config, callbacks=new_callback_manager)
    assert "run_name" in patched, (
        "run_name should be preserved when inherit_run_name=True"
    )
    assert patched["run_name"] == "parent_run", "run_name value should be unchanged"
    assert patched.get("inherit_run_name") is True, (
        "inherit_run_name should be preserved"
    )


def test_inherit_run_name_with_chain() -> None:
    """Test inherit_run_name behavior in a chain of runnables."""
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.runnables import RunnableLambda

    # Track run names through callbacks
    captured_names: list[str] = []

    class TestCallbackHandler(BaseCallbackHandler):
        def on_chain_start(self, serialized: dict, inputs: Any, **kwargs: Any) -> None:
            name = kwargs.get("name", "unnamed")
            captured_names.append(name)

    # Create a simple chain
    def identity(x: Any) -> Any:
        return x

    def process(x: Any) -> str:
        return f"processed: {x}"

    chain = RunnableLambda(identity) | RunnableLambda(process)

    # Test 1: Default behavior (run_name not inherited)
    captured_names.clear()
    config_default: RunnableConfig = {
        "run_name": "custom_chain_run",
        "callbacks": [TestCallbackHandler()],
    }
    result = chain.invoke("test", config=config_default)
    assert result == "processed: test"
    # First run should have custom name, child runs should have default names
    assert captured_names[0] == "custom_chain_run", "Root run should have custom name"
    # Child runs should NOT have the custom name (they'll have their default names)
    for i in range(1, len(captured_names)):
        assert captured_names[i] != "custom_chain_run", (
            f"Child run {i} should not inherit run_name by default"
        )

    # Test 2: With inherit_run_name=True
    captured_names.clear()
    config_inherit: RunnableConfig = {
        "run_name": "inherited_chain_run",
        "inherit_run_name": True,
        "callbacks": [TestCallbackHandler()],
    }
    result = chain.invoke("test", config=config_inherit)
    assert result == "processed: test"
    # All runs should have the same custom name when inherit_run_name=True
    assert captured_names[0] == "inherited_chain_run", (
        "Root run should have custom name"
    )
    # With inherit_run_name=True, child runs should also have the custom name
    for i in range(1, len(captured_names)):
        assert captured_names[i] == "inherited_chain_run", (
            f"Child run {i} should inherit run_name when inherit_run_name=True"
        )


def test_inherit_run_name_with_override() -> None:
    """Test that per-step with_config can still set different run_names when inherit_run_name is NOT used.

    This test verifies that the traditional behavior of setting different run_names
    per step via with_config still works when inherit_run_name is not enabled.
    """
    from typing import List

    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.runnables import RunnableLambda

    # Track run names through callbacks
    captured_names: List[str] = []

    class TestCallbackHandler(BaseCallbackHandler):
        def on_chain_start(self, serialized: dict, inputs: Any, **kwargs: Any) -> None:
            name = kwargs.get("name", "unnamed")
            captured_names.append(name)

    def identity(x: Any) -> Any:
        return x

    def process(x: Any) -> str:
        return f"processed: {x}"

    # Test: Without inherit_run_name, per-step with_config works as before
    chain = (
        RunnableLambda(identity).with_config(run_name="step1")
        | RunnableLambda(process).with_config(run_name="step2")
        | RunnableLambda(identity).with_config(run_name="step3")
    )

    captured_names.clear()
    config: RunnableConfig = {
        "run_name": "root_run",
        # Note: inherit_run_name is NOT set (defaults to False)
        "callbacks": [TestCallbackHandler()],
    }

    result = chain.invoke("test", config=config)
    assert result == "processed: test"

    # The root should have "root_run", and each step should have its own name
    assert "root_run" in captured_names, "Root run should have the config run_name"
    assert "step1" in captured_names, "First step should have its own run_name"
    assert "step2" in captured_names, "Second step should have its own run_name"
    assert "step3" in captured_names, "Third step should have its own run_name"


def test_inherit_run_name_merge_configs() -> None:
    """Test that inherit_run_name is properly handled in merge_configs."""
    from langchain_core.runnables.config import merge_configs

    # Test merging with inherit_run_name
    base: RunnableConfig = {
        "run_name": "base_run",
        "inherit_run_name": False,
    }

    override: RunnableConfig = {
        "inherit_run_name": True,
    }

    merged = merge_configs(base, override)
    assert merged.get("inherit_run_name") is True, (
        "inherit_run_name should be overridden"
    )
    assert merged.get("run_name") == "base_run", (
        "run_name should be preserved from base"
    )

    # Test that inherit_run_name passes through ensure_config
    from langchain_core.runnables.config import ensure_config

    config_with_inherit: RunnableConfig = {
        "run_name": "test_run",
        "inherit_run_name": True,
    }

    ensured = ensure_config(config_with_inherit)
    assert ensured.get("inherit_run_name") is True, (
        "inherit_run_name should pass through ensure_config"
    )
    assert ensured.get("run_name") == "test_run", "run_name should be preserved"

