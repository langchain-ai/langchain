"""Loader that loads Microsoft Excel files."""
from typing import Any, List

from langchain.document_loaders.unstructured import (
    UnstructuredFileLoader,
    validate_unstructured_version,
)


class UnstructuredXMLLoader(UnstructuredFileLoader):
    """Loader that uses unstructured to load XML files."""

    def __init__(
        self, file_path: str, mode: str = "single", **unstructured_kwargs: Any
    ):
        validate_unstructured_version(min_unstructured_version="0.6.7")
        super().__init__(file_path=file_path, mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List:
        from unstructured.partition.xml import partition_xml

        return partition_xml(filename=self.file_path, **self.unstructured_kwargs)
