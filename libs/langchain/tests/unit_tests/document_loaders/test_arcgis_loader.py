import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture
def arcgis_mocks(mock_feature_layer, mock_gis):
    sys_modules = {
        'arcgis_loader': MagicMock(),
        'arcgis.features.FeatureLayer': mock_feature_layer,
        'arcgis.gis.GIS': mock_gis,
    }
    with patch.dict('sys.modules', sys_modules):
        yield

@pytest.fixture
def mock_feature_layer():
    feature_layer = MagicMock()
    feature_layer.query.return_value = [MagicMock(as_dict={"attributes": {"field": "value"}})]
    feature_layer.url = "https://example.com/layer_url"
    feature_layer.properties = {"description": "<html><body>Some HTML content</body></html>"}
    return feature_layer


@pytest.fixture
def mock_gis():
    gis = MagicMock()
    gis.content.get.return_value = MagicMock(description="Item description")
    return gis


from langchain.document_loaders import ArcGISLoader

def test_lazy_load(arcgis_mocks, mock_feature_layer, mock_gis):
    loader = ArcGISLoader(layer=mock_feature_layer, gis=mock_gis)
    loader.BEAUTIFULSOUP = None

    documents = list(loader.lazy_load())

    assert len(documents) == 1
    assert documents[0].metadata['url'] == "https://example.com/layer_url"
    # Add more assertions based on your expected behavior
