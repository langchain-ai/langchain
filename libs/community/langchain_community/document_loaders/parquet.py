from typing import Any, Generator, List, Optional, Union

from langchain_core.documents.base import Document
from pyarrow.lib import Table
from pyarrow.parquet import ParquetFile

from langchain_community.document_loaders.base import BaseLoader


class ParquetLoader(BaseLoader):
    """
    Loads documents efficiently from Parquet files.

    This class provides a way to process Parquet files and extract documents,
    each representing a row in the file. It utilizes a chunking approach
    (row groups) for memory-efficient handling of large datasets.

    Attributes:
        file_path (str): Path to the Parquet file to be loaded.
        text_columns (List[str]): Columns containing text for the document's content.
        metadata_columns (Optional[List[str]]): Columns for the document's metadata
                                               (defaults to all non-text columns).
        parquetfile_kwargs (dict): Additional arguments passed to pyarrow's ParquetFile class.

    Methods:
        lazy_load(): Generator yielding Document objects for each row in the Parquet file.

    Example:
        Load documents specifying content and optional metadata columns:

        >>> loader = ParquetLoader("path/to/file.parquet",
        ...                         content_columns=["content", "summary"],
        ...                         metadata_columns=["title", "author"])
        >>> for doc in loader.lazy_load():
        ...     print(doc.page_content)
        ...     print(doc.metadata)

    Note:
        This loader leverages pyarrow for efficient Parquet file processing.
        It iterates through the file in chunks (row groups) to minimize memory usage
        when handling large datasets.
    """

    def __init__(
        self,
        file_path: str,
        content_columns: Union[str, List[str]],
        metadata_columns: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ):
        """
        Initializes the ParquetLoader instance.

        Args:
            file_path (str): Path to the Parquet file to be loaded.
            content_columns (List[str]): List of column names containing text for the document's content.
            metadata_columns (Optional[List[str]]): Optional list of column names for the document's metadata
                                                    (defaults to all non-text columns).
            **kwargs: Additional keyword arguments passed to pyarrow's ParquetFile class.

            Raises:
            TypeError if content_columns is not provided or is None
        """
        self.file_path = file_path

        self.content_columns = (
            [content_columns] if isinstance(content_columns, str) else content_columns
        )
        
        if metadata_columns:
            self.metadata_columns = (
                [metadata_columns]
                if isinstance(metadata_columns, str)
                else metadata_columns
            )
        else:
            self.metadata_columns = []
        self.parquetfile_kwargs = kwargs

        if not self.content_columns:
            raise TypeError("content_columns must be provided as a str or List[str]")

    def lazy_load(self) -> Generator[Document, Any, None]:
        """
        Lazily loads documents from a Parquet file.

        This method efficiently processes the Parquet file one document (row)
        at a time, minimizing memory usage.

        Yields:
            Document: A Document object for each row in the Parquet file. The
                    document's `page_content` is created by concatenating text
                    from specified columns. The document's `metadata` contains
                    either user-defined columns (`metadata_contents`) or all
                    remaining columns (excluding content_columns).

        Raises:
            Any exceptions raised by pyarrow when reading the Parquet file.

        Note:
            This method uses a generator to yield documents on-demand,
            avoiding loading the entire file into memory at once. This is
            particularly useful for handling large Parquet files.
        """
        parquet = ParquetFile(self.file_path, **self.parquetfile_kwargs)

        # To minimize memory footprint, iterate over the file by row_group and yield documents
        for row_group_num in range(parquet.num_row_groups):
            sub_table: Table = parquet.read_row_group(row_group_num)

            for row in sub_table.to_pylist():
                # Combine text from specified columns for Document.page_content
                text = " ".join(
                    str(row[col]) for col in self.content_columns if col in row
                )

                # Create metadata from other columns
                if self.metadata_columns:
                    metadata = {
                        col: row[col] for col in row if col in self.metadata_columns if col in row
                    }
                else:
                    metadata = {
                        col: row[col] for col in row if col not in self.content_columns
                    }

                # Create a Document object
                yield Document(page_content=text, metadata=metadata)
