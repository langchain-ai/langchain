"""Callback handlers that allow listening to events in LangChain."""
import os
from typing import Optional

import langchain
from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager
from langchain.callbacks.shared import SharedCallbackManager
from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.callbacks.tracers import SharedLangChainTracer


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
        session = os.environ.get("LANGCHAIN_SESSION")
        session = session if session is None else int(session)
        set_tracing_callback_manager(session)
    else:
        raise ValueError(
            f"LANGCHAIN_HANDLER should be one of `stdout` or `langchain`, got {default_handler}"
        )


def set_tracing_callback_manager(session: Optional[int] = None) -> None:
    """Set tracing callback manager."""
    handler = SharedLangChainTracer()
    callback = get_callback_manager()
    callback.set_handlers([handler, StdOutCallbackHandler()])
    if session is None:
        handler.load_default_session()
    else:
        try:
            handler.load_session(int(session))
        except:
            raise ValueError(f"session {session} not found")
