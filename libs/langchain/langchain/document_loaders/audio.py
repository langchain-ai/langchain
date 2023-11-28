from typing import Any, Iterator, List

from langchain_core.documents.base import Document

from langchain.document_loaders import Blob
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.parsers.audio import AzureSpeechServiceParser


class AzureSpeechServiceLoader(BaseLoader):
    def __init__(self, file_path: str, **kwargs: Any) -> None:
        """Initialize with file path."""
        super().__init__()

        self.file_path = file_path
        self.parser = AzureSpeechServiceParser(**kwargs)

    def load(self) -> List[Document]:
        """Eagerly load the content."""
        return list(self.lazy_load())

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Lazily lod documents."""
        blob = Blob.from_path(self.file_path)
        return iter(self.parser.parse(blob))
