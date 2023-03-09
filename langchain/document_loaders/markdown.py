"""Loader that loads Markdown files."""
from typing import List

from langchain.document_loaders.unstructured import UnstructuredFileLoader


class UnstructuredMarkdownLoader(UnstructuredFileLoader):
    """Loader that uses unstructured to load markdown files."""

    def _get_elements(self) -> List:
        from unstructured.__version__ import __version__ as __unstructured_version__
        from unstructured.partition.md import partition_md


        unstructured_version = tuple(
            [int(x) for x in __unstructured_version__.split(".")]
        )

        if unstructured_version < (0, 4, 16):
            raise ValueError(
                f"You are on unstructured version {__unstructured_version__}. "
                "Partitioning markdown files is only supported in unstructured>=0.4.11."
            )

        return partition_md(filename=self.file_path)
