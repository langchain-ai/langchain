from typing import TYPE_CHECKING, Any

from langchain_core.document_loaders import Blob, BlobLoader

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.document_loaders import (
        FileSystemBlobLoader,
        YoutubeAudioLoader,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "BlobLoader": "langchain_community.document_loaders",
    "Blob": "langchain_community.document_loaders",
    "FileSystemBlobLoader": "langchain_community.document_loaders",
    "YoutubeAudioLoader": "langchain_community.document_loaders",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "Blob",
    "BlobLoader",
    "FileSystemBlobLoader",
    "YoutubeAudioLoader",
]
