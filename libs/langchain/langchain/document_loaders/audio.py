from __future__ import annotations

from typing import Any, List

from langchain_core.documents import Document

from langchain.document_loaders import Blob
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.parsers.audio import AzureSpeechServiceParser


class AzureSpeechServiceLoader(BaseLoader):
    def load(self) -> List[Document]:
        blob = Blob.from_path(self.file_path)
        return self.parser.parse(blob)

    def __init__(self, file_path: str, **kwargs: Any) -> None:
        """
        Args:
            file_path: The path to the audio file.
        """
        self.file_path = file_path
        self.parser = AzureSpeechServiceParser(**kwargs)
