"""Loader that loads Microsoft Word files."""
from typing import List

from langchain.document_loaders.unstructured import UnstructuredFileLoader


class UnstructuredDocxLoader(UnstructuredFileLoader):
    """Loader that uses unstructured to load Microsoft Word files."""

    def _get_elements(self) -> List:
        from unstructured.partition.docx import partition_docx

        return partition_docx(filename=self.file_path)
