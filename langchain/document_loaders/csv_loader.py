import csv
from typing import Any, Dict, List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.unstructured import (
    UnstructuredFileLoader,
    validate_unstructured_version,
)


class CSVLoader(BaseLoader):
    """Loads a CSV file into a list of documents.

    Each document represents one row of the CSV file. Every row is converted into a
    key/value pair and outputted to a new line in the document's page_content.

    The source for each document loaded from csv is set to the value of the
    `file_path` argument for all doucments by default.
    You can override this by setting the `source_column` argument to the
    name of a column in the CSV file.
    The source of each document will then be set to the value of the column
    with the name specified in `source_column`.

    Output Example:
        .. code-block:: txt

            column1: value1
            column2: value2
            column3: value3
    """

    def __init__(
        self,
        file_path: str,
        source_column: Optional[str] = None,
        csv_args: Optional[Dict] = None,
        encoding: Optional[str] = None,
    ):
        self.file_path = file_path
        self.source_column = source_column
        self.encoding = encoding
        self.csv_args = csv_args or {}

    def load(self) -> List[Document]:
        """Load data into document objects."""

        docs = []
        with open(self.file_path, newline="", encoding=self.encoding) as csvfile:
            csv_reader = csv.DictReader(csvfile, **self.csv_args)  # type: ignore
            for i, row in enumerate(csv_reader):
                content = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in row.items())
                try:
                    source = (
                        row[self.source_column]
                        if self.source_column is not None
                        else self.file_path
                    )
                except KeyError:
                    raise ValueError(
                        f"Source column '{self.source_column}' not found in CSV file."
                    )
                metadata = {"source": source, "row": i}
                doc = Document(page_content=content, metadata=metadata)
                docs.append(doc)

        return docs


class UnstructuredCSVLoader(UnstructuredFileLoader):
    """Loader that uses unstructured to load CSV files."""

    def __init__(
        self, file_path: str, mode: str = "single", **unstructured_kwargs: Any
    ):
        validate_unstructured_version(min_unstructured_version="0.6.8")
        super().__init__(file_path=file_path, mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List:
        from unstructured.partition.csv import partition_csv

        return partition_csv(filename=self.file_path, **self.unstructured_kwargs)
