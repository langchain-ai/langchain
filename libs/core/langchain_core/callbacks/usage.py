"""Callback Handler that tracks `AIMessage.usage_metadata`."""

import threading
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from typing_extensions import override

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.messages.ai import UsageMetadata, add_usage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.tracers.context import register_configure_hook

# Module-level ContextVar registered once to prevent _configure_hooks accumulation.
# Each call to get_usage_metadata_callback() previously created a new ContextVar and
# registered a new hook, causing unbounded growth of _configure_hooks in long-running
# applications.
_usage_metadata_callback_var: ContextVar["UsageMetadataCallbackHandler | None"] = (
    ContextVar("usage_metadata_callback", default=None)
)
register_configure_hook(_usage_metadata_callback_var, inheritable=True)


class UsageMetadataCallbackHandler(BaseCallbackHandler):
    """Callback Handler that tracks `AIMessage.usage_metadata`.

    Example:
        ```python
        from langchain.chat_models import init_chat_model
        from langchain_core.callbacks import UsageMetadataCallbackHandler

        llm_1 = init_chat_model(model="openai:gpt-4o-mini")
        llm_2 = init_chat_model(model="anthropic:claude-haiku-4-5-20251001")

        callback = UsageMetadataCallbackHandler()
        result_1 = llm_1.invoke("Hello", config={"callbacks": [callback]})
        result_2 = llm_2.invoke("Hello", config={"callbacks": [callback]})
        callback.usage_metadata
        ```

        ```txt
        {'gpt-4o-mini-2024-07-18': {'input_tokens': 8,
          'output_tokens': 10,
          'total_tokens': 18,
          'input_token_details': {'audio': 0, 'cache_read': 0},
          'output_token_details': {'audio': 0, 'reasoning': 0}},
         'claude-haiku-4-5-20251001': {'input_tokens': 8,
          'output_tokens': 21,
          'total_tokens': 29,
          'input_token_details': {'cache_read': 0, 'cache_creation': 0}}}
        ```

    !!! version-added "Added in `langchain-core` 0.3.49"

    """

    def __init__(self) -> None:
        """Initialize the `UsageMetadataCallbackHandler`."""
        super().__init__()
        self._lock = threading.Lock()
        self.usage_metadata: dict[str, UsageMetadata] = {}

    @override
    def __repr__(self) -> str:
        return str(self.usage_metadata)

    @override
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
def get_usage_metadata_callback(
    name: str = "usage_metadata_callback",
) -> Generator[UsageMetadataCallbackHandler, None, None]:
    """Get usage metadata callback.

    Get context manager for tracking usage metadata across chat model calls using
    [`AIMessage.usage_metadata`][langchain.messages.AIMessage.usage_metadata].

    Args:
        name: Deprecated. This argument has no effect and will be removed in a
            future version. The underlying context variable is registered once at
            module load time to prevent unbounded growth of internal hook state.

    Yields:
        The usage metadata callback.

    Example:
        ```python
        from langchain.chat_models import init_chat_model
        from langchain_core.callbacks import get_usage_metadata_callback

        llm_1 = init_chat_model(model="openai:gpt-4o-mini")
        llm_2 = init_chat_model(model="anthropic:claude-haiku-4-5-20251001")

        with get_usage_metadata_callback() as cb:
            llm_1.invoke("Hello")
            llm_2.invoke("Hello")
            print(cb.usage_metadata)
        ```

        ```txt
        {
            "gpt-4o-mini-2024-07-18": {
                "input_tokens": 8,
                "output_tokens": 10,
                "total_tokens": 18,
                "input_token_details": {"audio": 0, "cache_read": 0},
                "output_token_details": {"audio": 0, "reasoning": 0},
            },
            "claude-haiku-4-5-20251001": {
                "input_tokens": 8,
                "output_tokens": 21,
                "total_tokens": 29,
                "input_token_details": {"cache_read": 0, "cache_creation": 0},
            },
        }
        ```

    !!! version-added "Added in `langchain-core` 0.3.49"

    """
    if name != "usage_metadata_callback":
        warnings.warn(
            "The `name` parameter of `get_usage_metadata_callback` is deprecated "
            "and will be removed in a future version. It has no effect.",
            DeprecationWarning,
            stacklevel=2,
        )
    cb = UsageMetadataCallbackHandler()
    token = _usage_metadata_callback_var.set(cb)
    try:
        yield cb
    finally:
        _usage_metadata_callback_var.reset(token)
