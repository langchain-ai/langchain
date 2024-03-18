"""**Index** is used to avoid writing duplicated content
into the vectostore and to avoid over-writing content if it's unchanged.

Indexes also :

* Create knowledge graphs from data.

* Support indexing workflows from LangChain data loaders to vectorstores.

Importantly, Index keeps on working even if the content being written is derived
via a set of transformations from some source content (e.g., indexing children
documents that were derived from parent documents by chunking.)
"""
from typing import Any

from langchain.indexes._api import IndexingResult, aindex, index
from langchain.indexes._sql_record_manager import SQLRecordManager
from langchain.indexes.vectorstore import VectorstoreIndexCreator

DEPRECATED_IMPORTS = [
    "GraphIndexCreator",
]


def __getattr__(name: str) -> Any:
    if name in DEPRECATED_IMPORTS:
        raise ImportError(
            f"{name} has been moved to the langchain-community package. "
            f"See https://github.com/langchain-ai/langchain/discussions/19083 for more "
            f"information.\n\nTo use it install langchain-community:\n\n"
            f"`pip install -U langchain-community`\n\n"
            f"then import with:\n\n"
            f"`from langchain_community.indexes.graph import {name}`"
        )
    raise AttributeError()


__all__ = [
    # Keep sorted
    "aindex",
    "index",
    "IndexingResult",
    "SQLRecordManager",
    "VectorstoreIndexCreator",
]
