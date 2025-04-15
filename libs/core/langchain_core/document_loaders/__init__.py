"""Document loaders."""

from importlib import import_module
from typing import TYPE_CHECKING

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
