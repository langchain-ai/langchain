"""Code to support various indexing workflows.

Provides code to:

* Create knowledge graphs from data.

* Support indexing workflows from LangChain data loaders to vectorstores.

For indexing workflows, this code is used to avoid writing duplicated content
into the vectostore and to avoid over-writing content if it's unchanged.

Importantly, this keeps on working even if the content being written is derived
via a set of transformations from some source content (e.g., indexing children
documents that were derived from parent documents by chunking.)
"""
from langchain.indexes._api import IndexingResult, index
from langchain.indexes._sql_record_manager import SQLRecordManager
from langchain.indexes.graph import GraphIndexCreator
from langchain.indexes.vectorstore import VectorstoreIndexCreator

__all__ = [
    # Keep sorted
    "GraphIndexCreator",
    "index",
    "IndexingResult",
    "SQLRecordManager",
    "VectorstoreIndexCreator",
]
