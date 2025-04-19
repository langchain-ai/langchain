"""Code to help indexing data into a vectorstore.

This package contains helper logic to help deal with indexing data into
a vectorstore while avoiding duplicated content and over-writing content
if it's unchanged.
"""

from typing import TYPE_CHECKING

from langchain_core._import_utils import import_attr

if TYPE_CHECKING:
    from langchain_core.indexing.api import IndexingResult, aindex, index
    from langchain_core.indexing.base import (
        DeleteResponse,
        DocumentIndex,
        InMemoryRecordManager,
        RecordManager,
        UpsertResponse,
    )

__all__ = (
    "aindex",
    "DeleteResponse",
    "DocumentIndex",
    "index",
    "IndexingResult",
    "InMemoryRecordManager",
    "RecordManager",
    "UpsertResponse",
)

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
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)
