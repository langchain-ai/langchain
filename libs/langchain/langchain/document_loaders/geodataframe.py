from typing import Any, Iterator, List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class GeoDataFrameLoader(BaseLoader):
    """Load `geopandas` Dataframe."""

    def __init__(self, data_frame: Any, page_content_column: str = "geometry"):
        """Initialize with geopandas Dataframe.

        Args:
            data_frame: geopandas DataFrame object.
            page_content_column: Name of the column containing the page content.
              Defaults to "geometry".
        """

        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError(
                "geopandas package not found, please install it with "
                "`pip install geopandas`"
            )

        if not isinstance(data_frame, gpd.GeoDataFrame):
            raise ValueError(
                f"Expected data_frame to be a gpd.GeoDataFrame, got {type(data_frame)}"
            )

        self.data_frame = data_frame
        self.page_content_column = page_content_column

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load records from dataframe."""

        for _, row in self.data_frame.iterrows():
            text = row[self.page_content_column]
            metadata = row.to_dict()
            metadata.pop(self.page_content_column)
            # Enforce str since shapely Point objects
            # geometry type used in GeoPandas) are not strings
            yield Document(page_content=str(text), metadata=metadata)

    def load(self) -> List[Document]:
        """Load full dataframe."""
        return list(self.lazy_load())
