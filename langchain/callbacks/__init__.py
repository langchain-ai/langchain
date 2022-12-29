"""Callback handlers that allow listening to events in LangChain."""
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.shared import SharedCallbackManager


def get_callback_manager() -> BaseCallbackManager:
    """Return the shared callback manager."""
    return SharedCallbackManager()
