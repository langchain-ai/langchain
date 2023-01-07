"""Callback handlers that allow listening to events in LangChain."""
import os
from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager
from langchain.callbacks.shared import SharedCallbackManager
from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.callbacks.tracers import SharedLangChainTracer
import langchain


def get_callback_manager() -> BaseCallbackManager:
    """Return the shared callback manager."""
    return SharedCallbackManager()


def set_handler(handler: BaseCallbackHandler) -> None:
    """Set handler."""
    callback = get_callback_manager()
    callback.set_handler(handler)


def set_default_callback_manager() -> None:
    """Set default callback manager."""
    default_handler = os.environ.get("LANGCHAIN_HANDLER", "stdout")
    if default_handler == "stdout":
        set_handler(StdOutCallbackHandler())
    elif default_handler == "langchain":
        langchain.verbose = True
        handler = SharedLangChainTracer()
        set_handler(handler)
        session = int(os.environ.get("LANGCHAIN_SESSION", "1"))
        try:
            handler.load_session(int(session))
        except:
            handler.new_session()
    else:
        raise ValueError(f"LANGCHAIN_HANDLER should be one of `stdout` or `langchain`, got {default_handler}")


def set_tracing_callback_manager() -> None:
    """Set tracing callback manager."""
    set_handler(SharedLangChainTracer())
