from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from langchain.document_loaders import GeoDataFrameLoader
from langchain.schema import Document

if TYPE_CHECKING:
    from geopandas import GeoDataFrame
else:
    GeoDataFrame = "geopandas.GeoDataFrame"


@pytest.mark.requires("geopandas")
def sample_gdf() -> GeoDataFrame:
    import geopandas

    # TODO: geopandas.datasets will be deprecated in 1.0
    path_to_data = geopandas.datasets.get_path("nybb")
    gdf = geopandas.read_file(path_to_data)
    gdf["area"] = gdf.area
    gdf["crs"] = gdf.crs.to_string()
    return gdf.head(2)


@pytest.mark.requires("geopandas")
def test_load_returns_list_of_documents(sample_gdf: GeoDataFrame) -> None:
    loader = GeoDataFrameLoader(sample_gdf)
    docs = loader.load()
    assert isinstance(docs, list)
    assert all(isinstance(doc, Document) for doc in docs)
    assert len(docs) == 2


@pytest.mark.requires("geopandas")
def test_load_converts_dataframe_columns_to_document_metadata(
    sample_gdf: GeoDataFrame,
) -> None:
    loader = GeoDataFrameLoader(sample_gdf)
    docs = loader.load()
    for i, doc in enumerate(docs):
        assert doc.metadata["area"] == sample_gdf.loc[i, "area"]
        assert doc.metadata["crs"] == sample_gdf.loc[i, "crs"]
