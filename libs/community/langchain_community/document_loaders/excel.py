"""Loads Microsoft Excel files."""

from pathlib import Path
from typing import Any, List, Union

from langchain_community.document_loaders.unstructured import (
    UnstructuredFileLoader,
    validate_unstructured_version,
)


class UnstructuredExcelLoader(UnstructuredFileLoader):
    """Load Microsoft Excel files using `Unstructured`.

    Like other
    Unstructured loaders, UnstructuredExcelLoader can be used in both
    "single" and "elements" mode. If you use the loader in "elements"
    mode, each sheet in the Excel file will be an Unstructured Table
    element. If you use the loader in "single" mode, an
    HTML representation of the table will be available in the
    "text_as_html" key in the document metadata.

    Examples
    --------
    from langchain_community.document_loaders.excel import UnstructuredExcelLoader

    loader = UnstructuredExcelLoader("stanley-cups.xlsx", mode="elements")
    docs = loader.load()
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        mode: str = "single",
        **unstructured_kwargs: Any,
    ):
        """

        Args:
            file_path: The path to the Microsoft Excel file.
            mode: The mode to use when partitioning the file. See unstructured docs
              for more info. Optional. Defaults to "single".
            **unstructured_kwargs: Keyword arguments to pass to unstructured.
        """
        validate_unstructured_version(min_unstructured_version="0.6.7")
        super().__init__(file_path=file_path, mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List:
        from unstructured.partition.xlsx import partition_xlsx

        return partition_xlsx(filename=self.file_path, **self.unstructured_kwargs)  # type: ignore[arg-type]
