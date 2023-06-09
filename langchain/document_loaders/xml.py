"""Loader that uses unstructured to load XML files."""
from typing import List

from langchain.document_loaders.unstructured import UnstructuredFileLoader


class UnstructuredXMLLoader(UnstructuredFileLoader):
    """Loader that uses unstructured to load XML files."""

    def _get_elements(self) -> List:
        from unstructured.partition.xml import partition_xml

        return partition_xml(filename=self.file_path, **self.unstructured_kwargs)
