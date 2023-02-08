"""Loader that loads Microsoft Word files."""
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class UnstructuredDocxLoader(BaseLoader):
    """Loader that uses unstructured to load Microsoft Word files."""

    def __init__(self, file_path: str):
        """Initialize with file path."""
        try:
            import unstructured  # noqa:F401
        except ImportError:
            raise ValueError(
                "unstructured package not found, please install it with "
                "`pip install unstructured`"
            )
        self.file_path = file_path

    def load(self) -> List[Document]:
        """Load file."""
        from unstructured.partition.docx import partition_docx

        elements = partition_docx(filename=self.file_path)
        text = "\n\n".join([str(el) for el in elements])
        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata)]
