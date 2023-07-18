import pytest

from langchain.document_loaders import GeoDataFrameLoader
from langchain.schema import Document

try:
    import geopandas

    GeoDataFrame = geopandas.GeoDataFrame
except ImportError:
    GeoDataFrame = None


requires_geopandas = pytest.mark.skipif(
    not pytest.importorskip("geopandas"), reason="geopandas is not installed"
)


@requires_geopandas
@pytest.fixture
def sample_gdf() -> GeoDataFrame:
    import geopandas

    path_to_data = geopandas.datasets.get_path("nybb")
    gdf = geopandas.read_file(path_to_data)
    gdf["area"] = gdf.area
    gdf["crs"] = gdf.crs.to_string()
    return gdf.head(2)


@requires_geopandas
def test_load_returns_list_of_documents(sample_gdf: GeoDataFrame) -> None:
    loader = GeoDataFrameLoader(sample_gdf)
    docs = loader.load()
    assert isinstance(docs, list)
    assert all(isinstance(doc, Document) for doc in docs)
    assert len(docs) == 2


@requires_geopandas
def test_load_converts_dataframe_columns_to_document_metadata(
    sample_gdf: GeoDataFrame,
) -> None:
    loader = GeoDataFrameLoader(sample_gdf)
    docs = loader.load()
    for i, doc in enumerate(docs):
        assert doc.metadata["area"] == sample_gdf.loc[i, "area"]
        assert doc.metadata["crs"] == sample_gdf.loc[i, "crs"]
