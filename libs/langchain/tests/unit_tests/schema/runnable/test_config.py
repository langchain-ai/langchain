from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.tracers.stdout import ConsoleCallbackHandler
from langchain.schema.runnable.config import merge_configs


def test_merge_config_callbacks() -> None:
    manager = CallbackManager(handlers=[StdOutCallbackHandler()])
    handlers = [ConsoleCallbackHandler()]

    merged = merge_configs({"callbacks": manager}, {"callbacks": handlers})["callbacks"]

    assert isinstance(merged, CallbackManager)
    assert len(merged.handlers) == 2
    assert isinstance(merged.handlers[0], StdOutCallbackHandler)
    assert isinstance(merged.handlers[1], ConsoleCallbackHandler)

    merged = merge_configs({"callbacks": handlers}, {"callbacks": manager})["callbacks"]

    assert isinstance(merged, CallbackManager)
    assert len(merged.handlers) == 2
    assert isinstance(merged.handlers[0], StdOutCallbackHandler)
    assert isinstance(merged.handlers[1], ConsoleCallbackHandler)

    merged = merge_configs(
        {"callbacks": handlers}, {"callbacks": [StreamingStdOutCallbackHandler()]}
    )["callbacks"]

    assert isinstance(merged, list)
    assert len(merged) == 2
    assert isinstance(merged[0], ConsoleCallbackHandler)
    assert isinstance(merged[1], StreamingStdOutCallbackHandler)
