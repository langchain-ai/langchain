from typing import Any, Dict, List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.unstructured import (
    UnstructuredFileLoader,
    validate_unstructured_version,
)


class UnstructuredTSVLoader(UnstructuredFileLoader):
    """Loader that uses unstructured to load TSV files."""

    def __init__(
        self, file_path: str, mode: str = "single", **unstructured_kwargs: Any
    ):
        validate_unstructured_version(min_unstructured_version="0.7.6")
        super().__init__(file_path=file_path, mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List:
        from unstructured.partition.tsv import partition_tsv

        return partition_tsv(filename=self.file_path, **self.unstructured_kwargs)
