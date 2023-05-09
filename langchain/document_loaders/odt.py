"""Loader that loads Open Office ODT files."""
from typing import Any, List

from langchain.document_loaders.unstructured import (
    UnstructuredFileLoader,
    validate_unstructured_version,
)


class UnstructuredODTLoader(UnstructuredFileLoader):
    """Loader that uses unstructured to load open office ODT files."""

    def __init__(
        self, file_path: str, mode: str = "single", **unstructured_kwargs: Any
    ):
        validate_unstructured_version(min_unstructured_version="0.6.3")
        super().__init__(file_path=file_path, mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List:
        from unstructured.partition.odt import partition_odt

        return partition_odt(filename=self.file_path, **self.unstructured_kwargs)
