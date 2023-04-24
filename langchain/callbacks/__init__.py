"""Callback handlers that allow listening to events in LangChain."""
import os
from contextlib import contextmanager
from typing import Generator, Optional

from langchain.callbacks.aim_callback import AimCallbackHandler
from langchain.callbacks.base import (
    BaseCallbackHandler,
    BaseCallbackManager,
)
from langchain.callbacks.clearml_callback import ClearMLCallbackHandler
from langchain.callbacks.comet_ml_callback import CometCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.openai_info import OpenAICallbackHandler
from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.wandb_callback import WandbCallbackHandler


@contextmanager
def get_openai_callback() -> Generator[OpenAICallbackHandler, None, None]:
    """Get OpenAI callback handler in a context manager."""
    handler = OpenAICallbackHandler()
    manager = CallbackManager([])
    manager.add_handler(handler)
    yield handler
    manager.remove_handler(handler)


__all__ = [
    "OpenAICallbackHandler",
    "StdOutCallbackHandler",
    "AimCallbackHandler",
    "WandbCallbackHandler",
    "ClearMLCallbackHandler",
    "CometCallbackHandler",
    "AsyncIteratorCallbackHandler",
    "get_openai_callback",
]
