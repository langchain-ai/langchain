"""Loader that loads rich text files."""
from typing import Any, List

from langchain.document_loaders.unstructured import (
    UnstructuredFileLoader,
    satisfies_min_unstructured_version,
)


class UnstructuredRTFLoader(UnstructuredFileLoader):
    """Loader that uses unstructured to load rtf files."""

    def __init__(
        self, file_path: str, mode: str = "single", **unstructured_kwargs: Any
    ):
        min_unstructured_version = "0.5.12"
        if not satisfies_min_unstructured_version(min_unstructured_version):
            raise ValueError(
                "Partitioning rtf files is only supported in "
                f"unstructured>={min_unstructured_version}."
            )

        super().__init__(file_path=file_path, mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List:
        from unstructured.partition.rtf import partition_rtf

        return partition_rtf(filename=self.file_path, **self.unstructured_kwargs)
