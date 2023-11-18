from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks.stdout import StdOutCallbackHandler
from langchain_core.callbacks.tracers.stdout import ConsoleCallbackHandler
from langchain_core.runnable.config import RunnableConfig, merge_configs


def test_merge_config_callbacks() -> None:
    manager: RunnableConfig = {
        "callbacks": CallbackManager(handlers=[StdOutCallbackHandler()])
    }
    handlers: RunnableConfig = {"callbacks": [ConsoleCallbackHandler()]}

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
