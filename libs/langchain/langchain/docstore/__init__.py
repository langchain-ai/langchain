"""DEPRECATED: This module has been moved to the langchain-community package.

**Docstores** are classes to store and load Documents."""
from typing import Any

DEPRECATED_IMPORTS = ["DocstoreFn", "InMemoryDocstore", "Wikipedia"]


def __getattr__(name: str) -> Any:
    if name in DEPRECATED_IMPORTS:
        raise ImportError(
            f"{name} has been moved to the langchain-community package. "
            f"See https://github.com/langchain-ai/langchain/discussions/19083 for more "
            f"information.\n\nTo use it install langchain-community:\n\n"
            f"`pip install -U langchain-community`\n\n"
            f"then import with:\n\n"
            f"`from langchain_community.docstore import {name}`"
        )

    raise AttributeError()
