"""Loader that loads Roam directory dump."""
from typing import List, Generator

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.blob_loaders.file_system import FileSystemLoader


class RoamLoader(BaseLoader):
    """Loader that loads Roam files from disk."""

    def __init__(self, path: str) -> None:
        """Initialize with path."""
        self.loader = FileSystemLoader(path, glob="**/*.md")

    def load(self) -> List[Document]:
        """Load documents."""
        return list(self.lazy_load())

    def lazy_load(
        self,
    ) -> Generator[Document, None, None]:
        """Load documents lazily."""
        for blob in self.loader.yield_blobs():
            yield Document(
                page_content=blob.as_string(), metadata={"source": blob.source}
            )
