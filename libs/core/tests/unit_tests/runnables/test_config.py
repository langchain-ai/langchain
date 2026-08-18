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
    _merge_metadata_dicts,
    _set_config_context,
    ensure_config,
    merge_configs,
    run_in_executor,
)
from langchain_core.tracers.stdout import ConsoleCallbackHandler

OPENAI_TEST_MODEL = "gpt-5.5"


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
                "model": OPENAI_TEST_MODEL,
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
        "model": OPENAI_TEST_MODEL,
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
        "model": OPENAI_TEST_MODEL,
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
                "model": OPENAI_TEST_MODEL,
                "metadata": {"nooverride": 18},
            },
        )
    )

    assert config["metadata"] == {"nooverride": 18, "model": OPENAI_TEST_MODEL}
    assert config["configurable"] == {"model": OPENAI_TEST_MODEL}


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


class TestMergeMetadataDicts:
    """Tests for _merge_metadata_dicts `lc_versions` merge behavior."""

    def test_lc_versions_preserves_both_nested_dicts(self) -> None:
        base = {"lc_versions": {"langchain-core": "0.3.1"}, "user_id": "abc"}
        incoming = {"lc_versions": {"langchain-anthropic": "1.3.3"}, "run": "x"}
        result = _merge_metadata_dicts(base, incoming)
        assert result == {
            "lc_versions": {
                "langchain-core": "0.3.1",
                "langchain-anthropic": "1.3.3",
            },
            "user_id": "abc",
            "run": "x",
        }

    def test_last_writer_wins_within_lc_versions(self) -> None:
        base = {"lc_versions": {"pkg": "1.0"}}
        incoming = {"lc_versions": {"pkg": "2.0"}}
        result = _merge_metadata_dicts(base, incoming)
        assert result == {"lc_versions": {"pkg": "2.0"}}

    def test_generic_nested_metadata_replaces(self) -> None:
        base = {"versions": {"app": "1.0"}, "nested": {"a": "1"}}
        incoming = {"versions": {"plugin": "2.0"}, "nested": {"b": "2"}}
        result = _merge_metadata_dicts(base, incoming)
        assert result == {"versions": {"plugin": "2.0"}, "nested": {"b": "2"}}

    def test_non_dict_overwrites_lc_versions_dict(self) -> None:
        base = {"lc_versions": {"nested": "value"}}
        incoming = {"lc_versions": "flat"}
        result = _merge_metadata_dicts(base, incoming)
        assert result == {"lc_versions": "flat"}

    def test_lc_versions_dict_overwrites_non_dict(self) -> None:
        base = {"lc_versions": "flat"}
        incoming = {"lc_versions": {"nested": "value"}}
        result = _merge_metadata_dicts(base, incoming)
        assert result == {"lc_versions": {"nested": "value"}}

    def test_no_mutation_of_lc_versions_inputs(self) -> None:
        base = {"lc_versions": {"a": "1"}}
        incoming = {"lc_versions": {"b": "2"}}
        base_copy = {"lc_versions": {"a": "1"}}
        incoming_copy = {"lc_versions": {"b": "2"}}
        result = _merge_metadata_dicts(base, incoming)
        assert base == base_copy
        assert incoming == incoming_copy
        assert result["lc_versions"] is not base["lc_versions"]
        assert result["lc_versions"] is not incoming["lc_versions"]

    def test_non_overlapping_lc_versions_dict_is_copied(self) -> None:
        base = {"lc_versions": {"a": "1"}, "extras": {"x": "y"}}
        result = _merge_metadata_dicts(base, {})
        assert result["lc_versions"] is not base["lc_versions"]
        assert result["lc_versions"] == {"a": "1"}
        assert result["extras"] is base["extras"]

    def test_both_empty(self) -> None:
        assert _merge_metadata_dicts({}, {}) == {}

    def test_empty_base(self) -> None:
        incoming = {"lc_versions": {"pkg": "1.0"}}
        result = _merge_metadata_dicts({}, incoming)
        assert result == {"lc_versions": {"pkg": "1.0"}}
        assert result["lc_versions"] is not incoming["lc_versions"]

        result["lc_versions"]["new"] = "2.0"
        assert incoming == {"lc_versions": {"pkg": "1.0"}}

    def test_empty_incoming(self) -> None:
        result = _merge_metadata_dicts({"lc_versions": {"pkg": "1.0"}}, {})
        assert result == {"lc_versions": {"pkg": "1.0"}}

    def test_merge_configs_with_none_metadata(self) -> None:
        merged = merge_configs(
            cast("RunnableConfig", {"metadata": None}),
            {"metadata": {"lc_versions": {"a": "1"}}},
        )
        assert merged["metadata"] == {"lc_versions": {"a": "1"}}

    def test_three_config_merge_accumulates_lc_versions(self) -> None:
        c1: RunnableConfig = {"metadata": {"lc_versions": {"a": "1"}}}
        c2: RunnableConfig = {"metadata": {"lc_versions": {"b": "2"}}}
        c3: RunnableConfig = {"metadata": {"lc_versions": {"c": "3"}}}
        merged = merge_configs(c1, c2, c3)
        assert merged["metadata"] == {
            "lc_versions": {"a": "1", "b": "2", "c": "3"},
        }
