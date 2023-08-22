"""Code to support indexing workflows with LangChain data loaders and vectorstores.

Use to:

1. Avoid writing duplicated content into the vectostore
2. Avoid over-writing content if it's unchanged

Importantly, this keeps on working even if the content being written is derived
via a set of transformations from some source content (e.g., indexing children
documents that were derived from parent documents by chunking.)
"""
from langchain.indexing._sql_record_manager import SQLRecordManager
from langchain.indexing.api import IndexingResult, index

__all__ = [
    # Keep sorted
    "index",
    "IndexingResult",
    "SQLRecordManager",
]
