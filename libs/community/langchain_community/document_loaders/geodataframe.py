from typing import Any, Iterator, List

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


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

        if page_content_column not in data_frame.columns:
            raise ValueError(
                f"Expected data_frame to have a column named {page_content_column}"
            )

        if not isinstance(data_frame[page_content_column], gpd.GeoSeries):
            raise ValueError(
                f"Expected data_frame[{page_content_column}] to be a GeoSeries"
            )

        self.data_frame = data_frame
        self.page_content_column = page_content_column

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load records from dataframe."""

        # assumes all geometries in GeoSeries are same CRS and Geom Type
        crs_str = self.data_frame.crs.to_string() if self.data_frame.crs else None
        geometry_type = self.data_frame.geometry.geom_type.iloc[0]

        for _, row in self.data_frame.iterrows():
            geom = row[self.page_content_column]

            xmin, ymin, xmax, ymax = geom.bounds

            metadata = row.to_dict()
            metadata["crs"] = crs_str
            metadata["geometry_type"] = geometry_type
            metadata["xmin"] = xmin
            metadata["ymin"] = ymin
            metadata["xmax"] = xmax
            metadata["ymax"] = ymax

            metadata.pop(self.page_content_column)

            # using WKT instead of str() to help GIS system interoperability
            yield Document(page_content=geom.wkt, metadata=metadata)

    def load(self) -> List[Document]:
        """Load full dataframe."""
        return list(self.lazy_load())
