"""Loads audio file transcription."""
import os
from typing import Iterator, List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.blob_loaders.file_system import FileSystemBlobLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers.audio import OpenAIWhisperParser


class OpenAIWhisperLoader(BaseLoader):
    """Loads audio file transcription using OpenAI Whisper"""

    def __init__(self, file_path: str, api_key: Optional[str] = None):
        """Initialize with file path."""
        self.file_path = file_path
        self.api_key = api_key
        if "~" in self.file_path:
            self.file_path = os.path.expanduser(self.file_path)

    def _get_loader(self) -> GenericLoader:
        return GenericLoader(
            FileSystemBlobLoader(
                os.path.dirname(self.file_path), glob=os.path.basename(self.file_path)
            ),
            OpenAIWhisperParser(self.api_key),
        )

    def load(self) -> List[Document]:
        """Load audio transcription into Document objects."""
        loader = self._get_loader()
        return loader.load()

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """A lazy loader for Documents."""
        loader = self._get_loader()
        yield from loader.lazy_load()
