"""Document loaders."""

from typing import TYPE_CHECKING

from langchain_core._import_utils import import_attr

if TYPE_CHECKING:
    from langchain_core.document_loaders.base import BaseBlobParser, BaseLoader
    from langchain_core.document_loaders.blob_loaders import Blob, BlobLoader, PathLike
    from langchain_core.document_loaders.langsmith import LangSmithLoader

__all__ = (
    "BaseBlobParser",
    "BaseLoader",
    "Blob",
    "BlobLoader",
    "PathLike",
    "LangSmithLoader",
)

_dynamic_imports = {
    "BaseBlobParser": "base",
    "BaseLoader": "base",
    "Blob": "blob_loaders",
    "BlobLoader": "blob_loaders",
    "PathLike": "blob_loaders",
    "LangSmithLoader": "langsmith",
}


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)
