"""Loader that loads PDF files."""
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.unstructured import UnstructuredFileLoader


class UnstructuredPDFLoader(UnstructuredFileLoader):
    """Loader that uses unstructured to load PDF files."""

    def _get_elements(self) -> List:
        from unstructured.partition.pdf import partition_pdf

        return partition_pdf(filename=self.file_path)

