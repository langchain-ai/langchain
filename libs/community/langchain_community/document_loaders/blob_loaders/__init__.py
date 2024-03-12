import importlib
from typing import Any

_module_lookup = {
    "Blob": "langchain_community.document_loaders.blob_loaders.schema",
    "BlobLoader": "langchain_community.document_loaders.blob_loaders.schema",
    "FileSystemBlobLoader": "langchain_community.document_loaders.blob_loaders.file_system",  # noqa: E501
    "YoutubeAudioLoader": "langchain_community.document_loaders.blob_loaders.youtube_audio",  # noqa: E501
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = list(_module_lookup.keys())
