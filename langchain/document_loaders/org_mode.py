"""Loader that loads Org-Mode files."""
from typing import Any, List

from langchain.document_loaders.unstructured import (
    UnstructuredFileLoader,
    validate_unstructured_version,
)


class UnstructuredOrgModeLoader(UnstructuredFileLoader):
    """Loader that uses unstructured to load Org-Mode files."""

    def __init__(
        self, file_path: str, mode: str = "single", **unstructured_kwargs: Any
    ):
        validate_unstructured_version(min_unstructured_version="0.7.9")
        super().__init__(file_path=file_path, mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List:
        from unstructured.partition.org import partition_org

        return partition_org(filename=self.file_path, **self.unstructured_kwargs)
