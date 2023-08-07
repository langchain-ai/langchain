from typing import Any, Iterator, List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class XorbitsLoader(BaseLoader):
    """Load Xorbits DataFrame."""

    def __init__(self, data_frame: Any, page_content_column: str = "text"):
        """Initialize with dataframe object.

        Requirements:
            Must have xorbits installed. You can install with `pip install xorbits`.

        Args:
            data_frame: Xorbits DataFrame object.
            page_content_column: Name of the column containing the page content.
              Defaults to "text".
        """
        try:
            import xorbits.pandas as pd
        except ImportError as e:
            raise ImportError(
                "Cannot import xorbits, please install with 'pip install xorbits'."
            ) from e

        if not isinstance(data_frame, pd.DataFrame):
            raise ValueError(
                f"Expected data_frame to be a xorbits.pandas.DataFrame, \
                  got {type(data_frame)}"
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
