from typing import Any, cast

import pytest

from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks.stdout import StdOutCallbackHandler
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.runnables import RunnableBinding, RunnablePassthrough
from langchain_core.runnables.config import (
    RunnableConfig,
    merge_configs,
    run_in_executor,
)
from langchain_core.tracers.stdout import ConsoleCallbackHandler


def test_merge_config_callbacks() -> None:
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


def test_config_arbitrary_keys() -> None:
    base: RunnablePassthrough[Any] = RunnablePassthrough()
    bound = base.with_config(my_custom_key="my custom value")
    config = cast(RunnableBinding, bound).config

    assert config.get("my_custom_key") == "my custom value"


async def test_run_in_executor() -> None:
    def raises_stop_iter() -> Any:
        return next(iter([]))

    with pytest.raises(StopIteration):
        raises_stop_iter()

    with pytest.raises(RuntimeError):
        await run_in_executor(None, raises_stop_iter)
