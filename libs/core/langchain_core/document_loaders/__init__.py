"""Document loaders."""

from typing import TYPE_CHECKING

from langchain_core._lazy_imports import create_dynamic_getattr

if TYPE_CHECKING:
    from langchain_core.document_loaders.base import BaseBlobParser, BaseLoader
    from langchain_core.document_loaders.blob_loaders import Blob, BlobLoader, PathLike
    from langchain_core.document_loaders.langsmith import LangSmithLoader

__all__ = [
    "BaseBlobParser",
    "BaseLoader",
    "Blob",
    "BlobLoader",
    "PathLike",
    "LangSmithLoader",
]

__getattr__ = create_dynamic_getattr(
    package_name="langchain_core",
    module_path="document_loaders",
    dynamic_imports={
        "BaseBlobParser": "base",
        "BaseLoader": "base",
        "Blob": "blob_loaders",
        "BlobLoader": "blob_loaders",
        "PathLike": "blob_loaders",
        "LangSmithLoader": "langsmith",
    },
)


def __dir__() -> list[str]:
    return list(__all__)
