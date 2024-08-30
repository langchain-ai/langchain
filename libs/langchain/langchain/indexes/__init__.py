"""**Index** is used to avoid writing duplicated content
into the vectostore and to avoid over-writing content if it's unchanged.

Indexes also :

* Create knowledge graphs from data.

* Support indexing workflows from LangChain data loaders to vectorstores.

Importantly, Index keeps on working even if the content being written is derived
via a set of transformations from some source content (e.g., indexing children
documents that were derived from parent documents by chunking.)
"""

from typing import TYPE_CHECKING, Any

from langchain_core.indexing.api import IndexingResult, aindex, index

from langchain._api import create_importer
from langchain.indexes._sql_record_manager import SQLRecordManager
from langchain.indexes.vectorstore import VectorstoreIndexCreator

if TYPE_CHECKING:
    from langchain_community.graphs.index_creator import GraphIndexCreator


# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "GraphIndexCreator": "langchain_community.graphs.index_creator",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    # Keep sorted
    "aindex",
    "GraphIndexCreator",
    "index",
    "IndexingResult",
    "SQLRecordManager",
    "VectorstoreIndexCreator",
]
