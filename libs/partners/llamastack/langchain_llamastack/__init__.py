"""
LangChain integration for Llama Stack.

This package provides LangChain-compatible interfaces for Llama Stack services,
including chat completion, embeddings, safety checking capabilities, and utilities.
"""

from langchain_llamastack._version import __version__
from langchain_llamastack.chat_models import ChatLlamaStack
from langchain_llamastack.embeddings import LlamaStackEmbeddings
from langchain_llamastack.safety import LlamaStackSafety
from langchain_llamastack.utils import (
    check_llamastack_connection,
    list_available_models,
)

__all__ = [
    "ChatLlamaStack",
    "LlamaStackEmbeddings",
    "LlamaStackSafety",
    "check_llamastack_connection",
    "list_available_models",
    "__version__",
]
