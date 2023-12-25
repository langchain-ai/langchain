from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.parsers.audio import AzureAISpeechParser


class AzureAISpeechLoader(BaseLoader):
    """Azure AI Speech Service Document Loader.

    A document loader that can load an audio file from the local file system
    and transcribe it using Azure AI Speech Service.

    Examples:

        .. code-block:: python

            from langchain_community.document_loaders import AzureAISpeechLoader

            loader = AzureAISpeechParser(
                file_path="path/to/directory/example.wav",
                api_key="speech-api-key-from-azure",
                region="speech-api-region-from-azure"
            )

            loader.lazy_load()
    """

    def load(self) -> List[Document]:
        blob = Blob.from_path(self.file_path)
        return self.parser.parse(blob)

    def lazy_load(self) -> List[Document]:
        return self.load()

    def __init__(self, file_path: str, **kwargs: Optional[list[str] | str]) -> None:
        """
        Args:
            file_path: The path to the audio file.
        """
        self.file_path = file_path
        self.parser = AzureAISpeechParser(**kwargs)
