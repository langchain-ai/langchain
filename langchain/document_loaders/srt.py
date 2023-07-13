"""Loader for .srt (subtitle) files."""
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class SRTLoader(BaseLoader):
    """Loader for .srt (subtitle) files."""

    def __init__(self, file_path: str):
        """Initialize with a file path."""
        try:
            import pysrt  # noqa:F401
        except ImportError:
            raise ImportError(
                "package `pysrt` not found, please install it with `pip install pysrt`"
            )
        self.file_path = file_path

    def load(self) -> List[Document]:
        """Load using pysrt file."""
        import pysrt

        parsed_info = pysrt.open(self.file_path)
        text = " ".join([t.text for t in parsed_info])
        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata)]
