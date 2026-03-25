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
    _merge_metadata_dicts,
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
    """Tests for _merge_metadata_dicts deep-merge behavior."""

    def test_deep_merge_preserves_both_nested_dicts(self) -> None:
        base = {"versions": {"langchain-core": "0.3.1"}, "user_id": "abc"}
        incoming = {"versions": {"langchain-anthropic": "1.3.3"}, "run": "x"}
        result = _merge_metadata_dicts(base, incoming)
        assert result == {
            "versions": {
                "langchain-core": "0.3.1",
                "langchain-anthropic": "1.3.3",
            },
            "user_id": "abc",
            "run": "x",
        }

    def test_last_writer_wins_within_nested_dicts(self) -> None:
        base = {"versions": {"pkg": "1.0"}}
        incoming = {"versions": {"pkg": "2.0"}}
        result = _merge_metadata_dicts(base, incoming)
        assert result == {"versions": {"pkg": "2.0"}}

    def test_non_dict_overwrites_dict(self) -> None:
        base = {"key": {"nested": "value"}}
        incoming = {"key": "flat"}
        result = _merge_metadata_dicts(base, incoming)
        assert result == {"key": "flat"}

    def test_dict_overwrites_non_dict(self) -> None:
        base = {"key": "flat"}
        incoming = {"key": {"nested": "value"}}
        result = _merge_metadata_dicts(base, incoming)
        assert result == {"key": {"nested": "value"}}

    def test_no_mutation_of_inputs(self) -> None:
        base = {"versions": {"a": "1"}}
        incoming = {"versions": {"b": "2"}}
        base_copy = {"versions": {"a": "1"}}
        incoming_copy = {"versions": {"b": "2"}}
        result = _merge_metadata_dicts(base, incoming)
        assert base == base_copy
        assert incoming == incoming_copy
        # Returned nested dicts should not share identity with originals.
        assert result["versions"] is not base["versions"]
        assert result["versions"] is not incoming["versions"]

    def test_non_overlapping_nested_dict_is_copied(self) -> None:
        base = {"versions": {"a": "1"}, "extras": {"x": "y"}}
        incoming = {"versions": {"b": "2"}}
        result = _merge_metadata_dicts(base, incoming)
        # "extras" was not in incoming — result should still be a copy.
        assert result["extras"] is not base["extras"]
        assert result["extras"] == {"x": "y"}

    def test_both_empty(self) -> None:
        assert _merge_metadata_dicts({}, {}) == {}

    def test_empty_base(self) -> None:
        result = _merge_metadata_dicts({}, {"versions": {"pkg": "1.0"}})
        assert result == {"versions": {"pkg": "1.0"}}

    def test_empty_incoming(self) -> None:
        result = _merge_metadata_dicts({"versions": {"pkg": "1.0"}}, {})
        assert result == {"versions": {"pkg": "1.0"}}

    def test_merge_configs_with_none_metadata(self) -> None:
        merged = merge_configs(
            cast("RunnableConfig", {"metadata": None}),
            {"metadata": {"versions": {"a": "1"}}},
        )
        assert merged["metadata"] == {"versions": {"a": "1"}}

    def test_three_config_merge_accumulates(self) -> None:
        c1: RunnableConfig = {"metadata": {"versions": {"a": "1"}}}
        c2: RunnableConfig = {"metadata": {"versions": {"b": "2"}}}
        c3: RunnableConfig = {"metadata": {"versions": {"c": "3"}}}
        merged = merge_configs(c1, c2, c3)
        assert merged["metadata"] == {
            "versions": {"a": "1", "b": "2", "c": "3"},
        }
