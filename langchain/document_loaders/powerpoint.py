"""Loader that loads powerpoint files."""
from typing import List

from langchain.document_loaders.unstructured import UnstructuredFileLoader


class UnstructuredPowerPointLoader(UnstructuredFileLoader):
    """Loader that uses unstructured to load powerpoint files."""

    def _get_elements(self) -> List:
        from unstructured.partition.pptx import partition_pptx

        return partition_pptx(filename=self.file_path)
