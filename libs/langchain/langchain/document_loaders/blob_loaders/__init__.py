from typing import TYPE_CHECKING, Any

from langchain_core.document_loaders import Blob, BlobLoader

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.document_loaders.blob_loaders.file_system import (
        FileSystemBlobLoader,
    )
    from langchain_community.document_loaders.blob_loaders.youtube_audio import (
        YoutubeAudioLoader,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "FileSystemBlobLoader": (
        "langchain_community.document_loaders.blob_loaders.file_system"
    ),
    "YoutubeAudioLoader": (
        "langchain_community.document_loaders.blob_loaders.youtube_audio"
    ),
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "BlobLoader",
    "Blob",
    "FileSystemBlobLoader",
    "YoutubeAudioLoader",
]
