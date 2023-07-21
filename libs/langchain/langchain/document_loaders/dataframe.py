"""Load from a Dataframe object"""
from typing import Any, Iterator, List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class DataFrameLoader(BaseLoader):
    """Load Pandas DataFrame."""

    def __init__(self, data_frame: Any, page_content_column: str = "text"):
        """Initialize with dataframe object.

        Args:
            data_frame: Pandas DataFrame object.
            page_content_column: Name of the column containing the page content.
              Defaults to "text".
        """
        import pandas as pd

        if not isinstance(data_frame, pd.DataFrame):
            raise ValueError(
                f"Expected data_frame to be a pd.DataFrame, got {type(data_frame)}"
            )
        self.data_frame = data_frame
        self.page_content_column = page_content_column

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load records from dataframe."""

        for _, row in self.data_frame.iterrows():
            text = row[self.page_content_column]
            metadata = row.to_dict()
            metadata.pop(self.page_content_column)
            yield Document(page_content=text, metadata=metadata)

    def load(self) -> List[Document]:
        """Load full dataframe."""
        return list(self.lazy_load())
