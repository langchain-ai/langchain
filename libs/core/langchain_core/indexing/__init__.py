"""Code to help indexing data into a vectorstore.

This package contains helper logic to help deal with indexing data into
a vectorstore while avoiding duplicated content and over-writing content
if it's unchanged.
"""

from typing import TYPE_CHECKING

from langchain_core._lazy_imports import create_dynamic_getattr

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

__getattr__ = create_dynamic_getattr(
    package_name="langchain_core",
    module_path="indexing",
    dynamic_imports={
        "aindex": "api",
        "index": "api",
        "IndexingResult": "api",
        "DeleteResponse": "base",
        "DocumentIndex": "base",
        "InMemoryRecordManager": "base",
        "RecordManager": "base",
        "UpsertResponse": "base",
    },
)


def __dir__() -> list[str]:
    return list(__all__)
