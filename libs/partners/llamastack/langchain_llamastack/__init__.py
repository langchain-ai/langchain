"""
LangChain integration for Llama Stack.

This package provides LangChain-compatible interfaces for Llama Stack services,
including chat completion, embeddings, safety checking capabilities, and utilities.
"""

from langchain_llamastack._version import __version__
from langchain_llamastack.chat_models import (
    check_llamastack_status,
    create_llamastack_llm,
    get_llamastack_models,
)
from langchain_llamastack.embeddings import LlamaStackEmbeddings
from langchain_llamastack.safety import LlamaStackSafety, SafetyResult
from langchain_llamastack.utils import (
    check_llamastack_connection,
    list_available_models,
)

__all__ = [
    # Factory functions
    "create_llamastack_llm",
    "get_llamastack_models",
    "check_llamastack_status",
    # Core classes
    "LlamaStackEmbeddings",
    "LlamaStackSafety",
    "SafetyResult",
    # Utility functions
    "check_llamastack_connection",
    "list_available_models",
    # Version
    "__version__",
]
