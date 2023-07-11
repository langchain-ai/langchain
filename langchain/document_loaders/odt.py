"""Loads OpenOffice ODT files."""
from typing import Any, List

from langchain.document_loaders.unstructured import (
    UnstructuredFileLoader,
    validate_unstructured_version,
)


class UnstructuredODTLoader(UnstructuredFileLoader):
    """Loader that uses unstructured to load OpenOffice ODT files."""

    def __init__(
        self, file_path: str, mode: str = "single", **unstructured_kwargs: Any
    ):
        """

        Args:
            file_path: The path to the file to load.
            mode: The mode to use when loading the file. Can be one of "single",
                "multi", or "all". Default is "single".
            **unstructured_kwargs: Any kwargs to pass to the unstructured.
        """
        validate_unstructured_version(min_unstructured_version="0.6.3")
        super().__init__(file_path=file_path, mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List:
        from unstructured.partition.odt import partition_odt

        return partition_odt(filename=self.file_path, **self.unstructured_kwargs)
