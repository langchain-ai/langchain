from langchain._api import create_importer

from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from langchain_community.vectorstores.sklearn import BaseSerializer
    from langchain_community.vectorstores.sklearn import JsonSerializer
    from langchain_community.vectorstores.sklearn import BsonSerializer
    from langchain_community.vectorstores.sklearn import ParquetSerializer
    from langchain_community.vectorstores.sklearn import SKLearnVectorStoreException
    from langchain_community.vectorstores import SKLearnVectorStore
            
# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {"BaseSerializer": "langchain_community.vectorstores.sklearn", "JsonSerializer": "langchain_community.vectorstores.sklearn", "BsonSerializer": "langchain_community.vectorstores.sklearn", "ParquetSerializer": "langchain_community.vectorstores.sklearn", "SKLearnVectorStoreException": "langchain_community.vectorstores.sklearn", "SKLearnVectorStore": "langchain_community.vectorstores"}
        
_import_attribute=create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)

def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)
__all__ = ["BaseSerializer",
"JsonSerializer",
"BsonSerializer",
"ParquetSerializer",
"SKLearnVectorStoreException",
"SKLearnVectorStore",
]
