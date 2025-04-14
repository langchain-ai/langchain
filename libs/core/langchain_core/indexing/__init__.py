"""Code to help indexing data into a vectorstore.

This package contains helper logic to help deal with indexing data into
a vectorstore while avoiding duplicated content and over-writing content
if it's unchanged.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.indexing.api import IndexingResult, aindex, index
    from langchain_core.indexing.base import (
        DeleteResponse,
        DocumentIndex,
        InMemoryRecordManager,
        RecordManager,
        UpsertResponse,
    )

__all__ = [
    "aindex",
    "DeleteResponse",
    "DocumentIndex",
    "index",
    "IndexingResult",
    "InMemoryRecordManager",
    "RecordManager",
    "UpsertResponse",
]

_dynamic_imports = {
    "aindex": "api",
    "index": "api",
    "IndexingResult": "api",
    "DeleteResponse": "base",
    "DocumentIndex": "base",
    "InMemoryRecordManager": "base",
    "RecordManager": "base",
    "UpsertResponse": "base",
}


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    package = __spec__.parent
    if module_name == "__module__" or module_name is None:
        result = import_module(f".{attr_name}", package=package)
    else:
        module = import_module(f".{module_name}", package=package)
        result = getattr(module, attr_name)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)
