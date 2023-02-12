"""Loader that uses unstructured to load files."""
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class UnstructuredFileLoader(BaseLoader):
    """Loader that uses unstructured to load files."""

    def __init__(self, file_path: str, mode: str = "single"):
        """Initialize with file path."""
        try:
            import unstructured  # noqa:F401
        except ImportError:
            raise ValueError(
                "unstructured package not found, please install it with "
                "`pip install unstructured`"
            )
        _valid_modes = {"single", "elements"}
        if mode not in _valid_modes:
            raise ValueError(
                f"Got {mode} for `mode`, but should be one of `{_valid_modes}`"
            )
        self.file_path = file_path
        self.mode = mode

    def _get_elements(self) -> List:
        from unstructured.partition.auto import partition

        return partition(filename=self.file_path)

    def load(self) -> List[Document]:
        """Load file."""
        elements = self._get_elements()
        metadata = {"source": self.file_path}
        if self.mode == "elements":
            docs = [
                Document(page_content=str(el), metadata=metadata) for el in elements
            ]
        elif self.mode == "single":
            text = "\n\n".join([str(el) for el in elements])
            docs = [Document(page_content=text, metadata=metadata)]
        else:
            raise ValueError(f"mode of {self.mode} not supported.")
        return docs
