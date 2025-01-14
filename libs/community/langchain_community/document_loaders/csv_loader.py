import csv
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.helpers import detect_file_encodings
from langchain_community.document_loaders.unstructured import (
    UnstructuredFileLoader,
    validate_unstructured_version,
)


class CSVLoader(BaseLoader):
    """Load a `CSV` file into a list of Documents.

    Each document represents one row of the CSV file. Every row is converted
    into a key/value pair and outputted to a new line in the document's
    page_content.

    The source for each document loaded from csv is set to the value of the
    `file_path` argument for all documents by default.
    You can override this by setting the `source_column` argument to the
    name of a column in the CSV file.
    The source of each document will then be set to the value of the column
    with the name specified in `source_column`.

    Output Example:
        .. code-block:: txt

            column1: value1
            column2: value2
            column3: value3

    Instantiate:
        .. code-block:: python

            from langchain_community.document_loaders import CSVLoader

            loader = CSVLoader(file_path='./hw_200.csv',
                csv_args={
                'delimiter': ',',
                'quotechar': '"',
                'fieldnames': ['Index', 'Height', 'Weight']
            })

    Load:
        .. code-block:: python

            docs = loader.load()
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            Index: Index
            Height: Height(Inches)"
            Weight: "Weight(Pounds)"
            {'source': './hw_200.csv', 'row': 0}

    Async load:
        .. code-block:: python

            docs = await loader.aload()
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            Index: Index
            Height: Height(Inches)"
            Weight: "Weight(Pounds)"
            {'source': './hw_200.csv', 'row': 0}

    Lazy load:
        .. code-block:: python

            docs = []
            docs_lazy = loader.lazy_load()

            # async variant:
            # docs_lazy = await loader.alazy_load()

            for doc in docs_lazy:
                docs.append(doc)
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            Index: Index
            Height: Height(Inches)"
            Weight: "Weight(Pounds)"
            {'source': './hw_200.csv', 'row': 0}
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        source_column: Optional[str] = None,
        metadata_columns: Sequence[str] = (),
        csv_args: Optional[Dict] = None,
        encoding: Optional[str] = None,
        autodetect_encoding: bool = False,
        *,
        content_columns: Sequence[str] = (),
    ):
        """

        Args:
            file_path: The path to the CSV file.
            source_column: The name of the column in the CSV file to use as the source.
              Optional. Defaults to None.
            metadata_columns: A sequence of column names to use as metadata. Optional.
            csv_args: A dictionary of arguments to pass to the csv.DictReader.
              Optional. Defaults to None.
            encoding: The encoding of the CSV file. Optional. Defaults to None.
            autodetect_encoding: Whether to try to autodetect the file encoding.
            content_columns: A sequence of column names to use for the document content.
                If not present, use all columns that are not part of the metadata.
        """
        self.file_path = file_path
        self.source_column = source_column
        self.metadata_columns = metadata_columns
        self.encoding = encoding
        self.csv_args = csv_args or {}
        self.autodetect_encoding = autodetect_encoding
        self.content_columns = content_columns

    def lazy_load(self) -> Iterator[Document]:
        try:
            with open(self.file_path, newline="", encoding=self.encoding) as csvfile:
                yield from self.__read_file(csvfile)
        except UnicodeDecodeError as e:
            if self.autodetect_encoding:
                detected_encodings = detect_file_encodings(self.file_path)
                for encoding in detected_encodings:
                    try:
                        with open(
                            self.file_path, newline="", encoding=encoding.encoding
                        ) as csvfile:
                            yield from self.__read_file(csvfile)
                            break
                    except UnicodeDecodeError:
                        continue
            else:
                raise RuntimeError(f"Error loading {self.file_path}") from e
        except Exception as e:
            raise RuntimeError(f"Error loading {self.file_path}") from e

    def __read_file(self, csvfile: TextIOWrapper) -> Iterator[Document]:
        csv_reader = csv.DictReader(csvfile, **self.csv_args)
        for i, row in enumerate(csv_reader):
            try:
                source = (
                    row[self.source_column]
                    if self.source_column is not None
                    else str(self.file_path)
                )
            except KeyError:
                raise ValueError(
                    f"Source column '{self.source_column}' not found in CSV file."
                )
            content = "\n".join(
                f"""{k.strip() if k is not None else k}: {
                    v.strip()
                    if isinstance(v, str)
                    else ",".join(map(str.strip, v))
                    if isinstance(v, list)
                    else v
                }"""
                for k, v in row.items()
                if (
                    k in self.content_columns
                    if self.content_columns
                    else k not in self.metadata_columns
                )
            )
            metadata = {"source": source, "row": i}
            for col in self.metadata_columns:
                try:
                    metadata[col] = row[col]
                except KeyError:
                    raise ValueError(f"Metadata column '{col}' not found in CSV file.")
            yield Document(page_content=content, metadata=metadata)


class UnstructuredCSVLoader(UnstructuredFileLoader):
    """Load `CSV` files using `Unstructured`.

    Like other
    Unstructured loaders, UnstructuredCSVLoader can be used in both
    "single" and "elements" mode. If you use the loader in "elements"
    mode, the CSV file will be a single Unstructured Table element.
    If you use the loader in "elements" mode, an HTML representation
    of the table will be available in the "text_as_html" key in the
    document metadata.

    Examples
    --------
    from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader

    loader = UnstructuredCSVLoader("stanley-cups.csv", mode="elements")
    docs = loader.load()
    """

    def __init__(
        self, file_path: str, mode: str = "single", **unstructured_kwargs: Any
    ):
        """

        Args:
            file_path: The path to the CSV file.
            mode: The mode to use when loading the CSV file.
              Optional. Defaults to "single".
            **unstructured_kwargs: Keyword arguments to pass to unstructured.
        """
        validate_unstructured_version(min_unstructured_version="0.6.8")
        super().__init__(file_path=file_path, mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List:
        from unstructured.partition.csv import partition_csv

        return partition_csv(filename=self.file_path, **self.unstructured_kwargs)  # type: ignore[arg-type]
