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

    assert config["metadata"] == {"nooverride": 18, "model": "gpt-4o"}
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
