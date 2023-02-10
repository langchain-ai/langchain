"""Loader that loads PDF files."""
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class UnstructuredURLLoader(BaseLoader):
    """Loader that uses unstructured to load HTML files."""

    def __init__(self, urls: List[str]):
        """Initialize with file path."""
        try:
            import unstructured  # noqa:F401
        except ImportError:
            raise ValueError(
                "unstructured package not found, please install it with "
                "`pip install unstructured`"
            )
        self.urls = urls

    def load(self) -> List[Document]:
        """Load file."""
        from unstructured.partition.html import partition_html

        docs: List[Document] = list()
        for url in self.urls:
            elements = partition_html(url=url)
            text = "\n\n".join([str(el) for el in elements])
            metadata = {"source": url}
            docs.append(Document(page_content=text, metadata=metadata))
        return docs
