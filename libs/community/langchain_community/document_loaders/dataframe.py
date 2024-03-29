from typing import Any, Iterator, Literal

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


class BaseDataFrameLoader(BaseLoader):
    def __init__(self, data_frame: Any, *, page_content_column: str = "text"):
        """Initialize with dataframe object.

        Args:
            data_frame: DataFrame object.
            page_content_column: Name of the column containing the page content.
              Defaults to "text".
        """
        self.data_frame = data_frame
        self.page_content_column = page_content_column

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load records from dataframe."""

        for _, row in self.data_frame.iterrows():
            text = row[self.page_content_column]
            metadata = row.to_dict()
            metadata.pop(self.page_content_column)
            yield Document(page_content=text, metadata=metadata)


class DataFrameLoader(BaseDataFrameLoader):
    """Load `Pandas` DataFrame."""

    def __init__(
        self,
        data_frame: Any,
        page_content_column: str = "text",
        engine: Literal["pandas", "modin"] = "pandas",
    ):
        """Initialize with dataframe object.

        Args:
            data_frame: Pandas DataFrame object.
            page_content_column: Name of the column containing the page content.
              Defaults to "text".
        """
        try:
            if engine == "pandas":
                import pandas as pd
            elif engine == "modin":
                import modin.pandas as pd
            else:
                raise ValueError(
                    f"Unsupported engine {engine}. Must be one of 'pandas', or 'modin'."
                )
        except ImportError as e:
            raise ImportError(
                "Unable to import pandas, please install with `pip install pandas`."
            ) from e

        if not isinstance(data_frame, pd.DataFrame):
            raise ValueError(
                f"Expected data_frame to be a pd.DataFrame, got {type(data_frame)}"
            )
        super().__init__(data_frame, page_content_column=page_content_column)
