"""Loader that uses unstructured to load HTML files."""
from typing import List

from langchain.document_loaders.unstructured import UnstructuredFileLoader


class UnstructuredHTMLLoader(UnstructuredFileLoader):
    """Loader that uses unstructured to load HTML files."""

    def _get_elements(self) -> List:
        from unstructured.partition.html import partition_html

        return partition_html(filename=self.file_path, **self.unstructured_kwargs)
