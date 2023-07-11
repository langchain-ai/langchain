"""Loads RST files."""
from typing import Any, List

from langchain.document_loaders.unstructured import (
    UnstructuredFileLoader,
    validate_unstructured_version,
)


class UnstructuredRSTLoader(UnstructuredFileLoader):
    """Loader that uses unstructured to load RST files."""

    def __init__(
        self, file_path: str, mode: str = "single", **unstructured_kwargs: Any
    ):
        """
        Initialize with a file path.

        Args:
            file_path: The path to the file to load.
            mode: The mode to use for partitioning. See unstructured for details.
                Defaults to "single".
            **unstructured_kwargs: Additional keyword arguments to pass
                to unstructured.
        """
        validate_unstructured_version(min_unstructured_version="0.7.5")
        super().__init__(file_path=file_path, mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List:
        from unstructured.partition.rst import partition_rst

        return partition_rst(filename=self.file_path, **self.unstructured_kwargs)
