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
    _get_langsmith_inheritable_metadata_from_config,
    _set_config_context,
    ensure_config,
    merge_configs,
    run_in_executor,
)
from langchain_core.tracers.stdout import ConsoleCallbackHandler


def test_ensure_config() -> None:
    run_id = str(uuid.uuid4())
    arg: dict[str, Any] = {
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
        "metadata": {"foo": "bar"},
        "callbacks": [arg["callbacks"][0]],
        "recursion_limit": 100,
        "configurable": {"baz": "qux", "something": "else"},
        "max_concurrency": 1,
        "run_id": run_id,
        "run_name": "test",
    }


def test_ensure_config_copies_model_to_metadata() -> None:
    config = ensure_config(
        {
            "configurable": {
                "thread_id": "th-123",
                "checkpoint_id": "ckpt-1",
                "checkpoint_ns": "ns-1",
                "task_id": "task-1",
                "run_id": "run-456",
                "assistant_id": "asst-789",
                "graph_id": "graph-0",
                "model": "gpt-4o",
                "user_id": "uid-1",
                "cron_id": "cron-1",
                "langgraph_auth_user_id": "user-1",
                "some_api_key": "opaque-token",
                "custom_setting": {"nested": True},
                "none_value": None,
            },
            "metadata": {"nooverride": 18},
        }
    )

    assert config["metadata"] == {
        "nooverride": 18,
        "model": "gpt-4o",
        "checkpoint_ns": "ns-1",
    }
    assert config["configurable"] == {
        "thread_id": "th-123",
        "checkpoint_id": "ckpt-1",
        "checkpoint_ns": "ns-1",
        "task_id": "task-1",
        "run_id": "run-456",
        "assistant_id": "asst-789",
        "graph_id": "graph-0",
        "model": "gpt-4o",
        "user_id": "uid-1",
        "cron_id": "cron-1",
        "langgraph_auth_user_id": "user-1",
        "some_api_key": "opaque-token",
        "custom_setting": {"nested": True},
        "none_value": None,
    }


def test_ensure_config_metadata_is_not_overridden_by_configurable_model() -> None:
    config = ensure_config(
        {
            "configurable": {
                "model": "from-configurable",
                "run_id": None,
                "checkpoint_ns": "from-configurable",
            },
            "metadata": {
                "model": "from-metadata",
                "run_id": "from-metadata",
                "checkpoint_ns": "from-metadata",
            },
        }
    )

    assert config["metadata"] == {
        "model": "from-metadata",
        "run_id": "from-metadata",
        "checkpoint_ns": "from-metadata",
    }
    assert config["configurable"] == {
        "model": "from-configurable",
        "run_id": None,
        "checkpoint_ns": "from-configurable",
    }


def test_ensure_config_copies_top_level_model_to_metadata() -> None:
    config = ensure_config(
        cast(
            "RunnableConfig",
            {
                "model": "gpt-4o",
                "metadata": {"nooverride": 18},
            },
        )
    )

    assert config["metadata"] == {"nooverride": 18, "model": "gpt-4o"}
    assert config["configurable"] == {"model": "gpt-4o"}


def test_ensure_config_copies_top_level_checkpoint_ns_to_metadata() -> None:
    config = ensure_config(
        cast(
            "RunnableConfig",
            {
                "checkpoint_ns": "ns-1",
                "metadata": {"nooverride": 18},
            },
        )
    )

    assert config["metadata"] == {"nooverride": 18, "checkpoint_ns": "ns-1"}
    assert config["configurable"] == {"checkpoint_ns": "ns-1"}


def test_get_langsmith_inheritable_metadata_from_config_uses_previous_copy_rules() -> (
    None
):
    config = ensure_config(
        cast(
            "RunnableConfig",
            {
                "something": "else",
                "metadata": {
                    "foo": "bar",
                    "model": "from-metadata",
                    "checkpoint_ns": "from-metadata",
                },
                "configurable": {
                    "baz": "qux",
                    "thread_id": "th-123",
                    "checkpoint_id": "ckpt-1",
                    "checkpoint_ns": "from-configurable",
                    "task_id": "task-1",
                    "run_id": "run-456",
                    "assistant_id": "asst-789",
                    "graph_id": "graph-0",
                    "model": "from-configurable",
                    "user_id": "uid-1",
                    "cron_id": "cron-1",
                    "langgraph_auth_user_id": "user-1",
                    "api_key": "should-not-propagate",
                    "__secret_key": "should-not-propagate",
                    "temperature": 0.5,
                    "streaming": True,
                    "custom_setting": {"nested": True},
                    "none_value": None,
                },
            },
        )
    )

    assert _get_langsmith_inheritable_metadata_from_config(config) == {
        "something": "else",
        "baz": "qux",
        "thread_id": "th-123",
        "checkpoint_id": "ckpt-1",
        "task_id": "task-1",
        "run_id": "run-456",
        "assistant_id": "asst-789",
        "graph_id": "graph-0",
        "user_id": "uid-1",
        "cron_id": "cron-1",
        "langgraph_auth_user_id": "user-1",
        "temperature": 0.5,
        "streaming": True,
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
    config = cast("RunnableBinding[Any, Any]", bound).config

    assert config.get("my_custom_key") == "my custom value"


async def test_run_in_executor() -> None:
    def raises_stop_iter() -> Any:
        return next(iter([]))

    with pytest.raises(StopIteration):
        raises_stop_iter()

    with pytest.raises(RuntimeError):
        await run_in_executor(None, raises_stop_iter)


# ---------------------------------------------------------------------------
# Regression tests for issue #37373
# ---------------------------------------------------------------------------


def test_configurable_data_included_in_metadata() -> None:
    """
    Regression test for issue #37373.
    Allowlisted configurable keys (model, checkpoint_ns) are automatically
    included in metadata so that callbacks and monitoring tools can access them.
    """
    from langchain_core.runnables import RunnableLambda

    received_config: RunnableConfig = {}

    def capture_config(x: Any, config: RunnableConfig) -> Any:
        received_config.update(config)
        return x

    runnable = RunnableLambda(capture_config)

    config = cast(
        "RunnableConfig",
        {
            "configurable": {"model": "gpt-4o", "checkpoint_ns": "ns-1"},
            "metadata": {"source": "test"},
        },
    )

    runnable.invoke("test", config=config)

    # Verify allowlisted configurable keys are propagated to metadata
    metadata = received_config.get("metadata", {})
    assert "model" in metadata, (
        "model should be copied from configurable into metadata"
    )
    assert metadata["model"] == "gpt-4o"
    assert "checkpoint_ns" in metadata, (
        "checkpoint_ns should be copied from configurable into metadata"
    )
    assert metadata["checkpoint_ns"] == "ns-1"

    # Verify original metadata is preserved alongside the propagated keys
    assert metadata["source"] == "test", "Original metadata should be preserved"


def test_metadata_not_overwritten_by_configurable() -> None:
    """
    Verify that if an allowlisted key exists in both metadata and configurable,
    the original metadata value takes precedence over the configurable value.
    """
    from langchain_core.runnables import RunnableLambda

    received_config: RunnableConfig = {}

    def capture_config(x: Any, config: RunnableConfig) -> Any:
        received_config.update(config)
        return x

    runnable = RunnableLambda(capture_config)

    config = cast(
        "RunnableConfig",
        {
            "configurable": {
                "model": "from-configurable",
                "checkpoint_ns": "from-configurable",
            },
            "metadata": {
                "model": "from-metadata",        # Should take precedence
                "checkpoint_ns": "from-metadata",  # Should take precedence
            },
        },
    )

    runnable.invoke("test", config=config)

    metadata = received_config.get("metadata", {})
    assert metadata["model"] == "from-metadata", (
        "Explicit metadata value should take precedence over configurable"
    )
    assert metadata["checkpoint_ns"] == "from-metadata", (
        "Explicit metadata value should take precedence over configurable"
    )


def test_non_allowlisted_configurable_keys_not_in_metadata() -> None:
    """
    Verify that arbitrary configurable keys (user_id, session_id, api_key, etc.)
    are NOT automatically propagated to metadata — only the allowlisted keys are.
    """
    from langchain_core.runnables import RunnableLambda

    received_config: RunnableConfig = {}

    def capture_config(x: Any, config: RunnableConfig) -> Any:
        received_config.update(config)
        return x

    runnable = RunnableLambda(capture_config)

    config = cast(
        "RunnableConfig",
        {
            "configurable": {
                "user_id": "alice",
                "session_id": "123",
                "some_api_key": "secret",
            },
            "metadata": {"source": "test"},
        },
    )

    runnable.invoke("test", config=config)

    metadata = received_config.get("metadata", {})
    # Non-allowlisted keys must NOT leak into metadata
    assert "user_id" not in metadata, (
        "user_id is not allowlisted and should not appear in metadata"
    )
    assert "session_id" not in metadata, (
        "session_id is not allowlisted and should not appear in metadata"
    )
    assert "some_api_key" not in metadata, (
        "some_api_key is not allowlisted and should not appear in metadata"
    )
    # Original metadata should still be intact
    assert metadata["source"] == "test", "Original metadata should be preserved"
