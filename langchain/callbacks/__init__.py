"""Callback handlers that allow listening to events in LangChain."""
import os
from contextlib import contextmanager
from typing import Generator, Optional

from langchain.callbacks.aim_callback import AimCallbackHandler
from langchain.callbacks.base import (
    BaseCallbackHandler,
    BaseCallbackManager,
    CallbackManager,
)
from langchain.callbacks.clearml_callback import ClearMLCallbackHandler
from langchain.callbacks.openai_info import OpenAICallbackHandler
from langchain.callbacks.shared import SharedCallbackManager
from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.callbacks.tracers import SharedLangChainTracer
from langchain.callbacks.wandb_callback import WandbCallbackHandler


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
        set_tracing_callback_manager(session)
    else:
        raise ValueError(
            f"LANGCHAIN_HANDLER should be one of `stdout` "
            f"or `langchain`, got {default_handler}"
        )


def set_tracing_callback_manager(session_name: Optional[str] = None) -> None:
    """Set tracing callback manager."""
    handler = SharedLangChainTracer()
    callback = get_callback_manager()
    callback.set_handlers([handler, StdOutCallbackHandler()])
    if session_name is None:
        handler.load_default_session()
    else:
        try:
            handler.load_session(session_name)
        except Exception:
            raise ValueError(f"session {session_name} not found")


@contextmanager
def get_openai_callback() -> Generator[OpenAICallbackHandler, None, None]:
    """Get OpenAI callback handler in a context manager."""
    handler = OpenAICallbackHandler()
    manager = get_callback_manager()
    manager.add_handler(handler)
    yield handler
    manager.remove_handler(handler)


__all__ = [
    "CallbackManager",
    "OpenAICallbackHandler",
    "SharedCallbackManager",
    "StdOutCallbackHandler",
    "AimCallbackHandler",
    "WandbCallbackHandler",
    "ClearMLCallbackHandler",
    "get_openai_callback",
    "set_tracing_callback_manager",
    "set_default_callback_manager",
    "set_handler",
    "get_callback_manager",
]
