from __future__ import annotations

from typing import Any

from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers.audio import AzureSpeechServiceParser


class AzureSpeechServiceLoader(GenericLoader):
    """Azure Speech Service Document Loader.

    A document loader that can load audio files from the local file system
    and transcribe them using Azure Speech Service.

    Examples:

        .. code-block:: python

            from langchain.document_loaders import AzureSpeechServiceLoader

            loader = AzureSpeechServiceLoader.from_filesystem(
                path="path/to/directory",
                glob="**/[!.]*",
                suffixes=[".wav"],
                show_progress=True,
            )

            loader.lazy_load()
    """

    @staticmethod
    def get_parser(**kwargs: Any) -> BaseBlobParser:
        """Get a parser for Azure Speech Service."""
        return AzureSpeechServiceParser(**kwargs)
