from typing import Any, Iterator, List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class PolarsDataFrameLoader(BaseLoader):
    """Load `Polars` DataFrame."""

    def __init__(self, data_frame: Any, page_content_column: str = "text"):
        """Initialize with dataframe object.

        Args:
            data_frame: Polars DataFrame object.
            page_content_column: Name of the column containing the page content.
              Defaults to "text".
        """
        import polars as pl

        if not isinstance(data_frame, pl.DataFrame):
            raise ValueError(
                f"Expected data_frame to be a pl.DataFrame, got {type(data_frame)}"
            )
        self.data_frame = data_frame
        self.page_content_column = page_content_column

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load records from dataframe."""

        for row in self.data_frame.iter_rows(named=True):
            text = row[self.page_content_column]
            row.pop(self.page_content_column)
            yield Document(page_content=text, metadata=row)

    def load(self) -> List[Document]:
        """Load full dataframe."""
        return list(self.lazy_load())
