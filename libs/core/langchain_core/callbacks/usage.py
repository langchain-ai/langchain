"""Callback Handler that tracks AIMessage.usage_metadata."""

import threading
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Optional

from langchain_core._api import beta
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.messages.ai import UsageMetadata, add_usage
from langchain_core.outputs import ChatGeneration, LLMResult


@beta()
class UsageMetadataCallbackHandler(BaseCallbackHandler):
    """Callback Handler that tracks AIMessage.usage_metadata.

    Example:
        .. code-block:: python

            from langchain.chat_models import init_chat_model
            from langchain_core.callbacks import UsageMetadataCallbackHandler

            llm = init_chat_model(model="openai:gpt-4o-mini")

            callback = UsageMetadataCallbackHandler()
            results = llm.batch(["Hello", "Goodbye"], config={"callbacks": [callback]})
            print(callback.usage_metadata)

        .. code-block:: none

            {'output_token_details': {'audio': 0, 'reasoning': 0}, 'input_tokens': 17, 'output_tokens': 31, 'total_tokens': 48, 'input_token_details': {'cache_read': 0, 'audio': 0}}

    .. versionadded:: 0.3.49
    """  # noqa: E501

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.usage_metadata: dict[str, UsageMetadata] = {}

    def __repr__(self) -> str:
        return str(self.usage_metadata)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Collect token usage."""
        # Check for usage_metadata (langchain-core >= 0.2.2)
        try:
            generation = response.generations[0][0]
        except IndexError:
            generation = None

        usage_metadata = None
        model_name = None
        if isinstance(generation, ChatGeneration):
            try:
                message = generation.message
                if isinstance(message, AIMessage):
                    usage_metadata = message.usage_metadata
                    model_name = message.response_metadata.get("model_name")
            except AttributeError:
                pass

        # update shared state behind lock
        if usage_metadata and model_name:
            with self._lock:
                if model_name not in self.usage_metadata:
                    self.usage_metadata[model_name] = usage_metadata
                else:
                    self.usage_metadata[model_name] = add_usage(
                        self.usage_metadata[model_name], usage_metadata
                    )


@contextmanager
@beta()
def get_usage_metadata_callback(
    name: str = "usage_metadata_callback",
) -> Generator[UsageMetadataCallbackHandler, None, None]:
    """Get context manager for tracking usage metadata across chat model calls using
    ``AIMessage.usage_metadata``.

    Args:
        name (str): The name of the context variable. Defaults to
            ``"usage_metadata_callback"``.

    Example:
        .. code-block:: python

            from langchain.chat_models import init_chat_model
            from langchain_core.callbacks import get_usage_metadata_callback

            llm = init_chat_model(model="openai:gpt-4o-mini")

            with get_usage_metadata_callback() as cb:
                llm.invoke("Hello")
                llm.invoke("Goodbye")
                print(cb.usage_metadata)

        .. code-block:: none

            {'output_token_details': {'audio': 0, 'reasoning': 0}, 'input_tokens': 17, 'output_tokens': 31, 'total_tokens': 48, 'input_token_details': {'cache_read': 0, 'audio': 0}}

    .. versionadded:: 0.3.49
    """  # noqa: E501
    from langchain_core.tracers.context import register_configure_hook

    usage_metadata_callback_var: ContextVar[Optional[UsageMetadataCallbackHandler]] = (
        ContextVar(name, default=None)
    )
    register_configure_hook(usage_metadata_callback_var, True)
    cb = UsageMetadataCallbackHandler()
    usage_metadata_callback_var.set(cb)
    yield cb
    usage_metadata_callback_var.set(None)
