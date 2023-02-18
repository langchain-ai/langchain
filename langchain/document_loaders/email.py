"""Loader that loads email files."""
from typing import List

from langchain.document_loaders.unstructured import UnstructuredFileLoader


class UnstructuredEmailLoader(UnstructuredFileLoader):
    """Loader that uses unstructured to load email files."""

    def _get_elements(self) -> List:
        from unstructured.partition.email import partition_email

        return partition_email(filename=self.file_path)
