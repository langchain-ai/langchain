"""Loader that loads EPub files."""
from typing import List

from langchain.document_loaders.unstructured import (
    UnstructuredFileLoader,
    satisfies_min_unstructured_version,
)


class UnstructuredEPubLoader(UnstructuredFileLoader):
    """Loader that uses unstructured to load epub files."""

    def _get_elements(self) -> List:
        min_unstructured_version = "0.5.4"
        if not satisfies_min_unstructured_version(min_unstructured_version):
            raise ValueError(
                "Partitioning epub files is only supported in "
                f"unstructured>={min_unstructured_version}."
            )
        from unstructured.partition.epub import partition_epub

        return partition_epub(filename=self.file_path)
