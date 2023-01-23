"""Callback handlers that allow listening to events in LangChain."""
from contextlib import contextmanager
from typing import Generator

from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager
from langchain.callbacks.openai_info import OpenAICallbackHandler
from langchain.callbacks.shared import SharedCallbackManager
from langchain.callbacks.stdout import StdOutCallbackHandler


def get_callback_manager() -> BaseCallbackManager:
    """Return the shared callback manager."""
    return SharedCallbackManager()


def set_handler(handler: BaseCallbackHandler) -> None:
    """Set handler."""
    callback = get_callback_manager()
    callback.set_handler(handler)


def set_default_callback_manager() -> None:
    """Set default callback manager."""
    set_handler(StdOutCallbackHandler())


@contextmanager
def get_openai_callback() -> Generator[OpenAICallbackHandler, None, None]:
    """Get OpenAI callback handler in a context manager."""
    handler = OpenAICallbackHandler()
    manager = get_callback_manager()
    manager.add_handler(handler)
    yield handler
    manager.remove_handler(handler)
