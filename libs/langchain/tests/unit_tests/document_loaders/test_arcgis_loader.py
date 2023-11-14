from unittest.mock import MagicMock, patch

import pytest

from langchain.document_loaders import ArcGISLoader


@pytest.fixture
def arcgis_mocks(mock_feature_layer, mock_gis):  # type: ignore
    sys_modules = {
        "arcgis": MagicMock(),
        "arcgis.features.FeatureLayer": mock_feature_layer,
        "arcgis.gis.GIS": mock_gis,
    }
    with patch.dict("sys.modules", sys_modules):
        yield


@pytest.fixture
def mock_feature_layer():  # type: ignore
    feature_layer = MagicMock()
    feature_layer.query.return_value = [
        MagicMock(as_dict={"attributes": {"field": "value"}})
    ]
    feature_layer.url = "https://example.com/layer_url"
    feature_layer.properties = {
        "description": "<html><body>Some HTML content</body></html>",
        "name": "test",
        "serviceItemId": "testItemId",
    }
    return feature_layer


@pytest.fixture
def mock_gis():  # type: ignore
    gis = MagicMock()
    gis.content.get.return_value = MagicMock(description="Item description")
    return gis


def test_lazy_load(arcgis_mocks, mock_feature_layer, mock_gis):  # type: ignore
    loader = ArcGISLoader(layer=mock_feature_layer, gis=mock_gis)
    loader.BEAUTIFULSOUP = None

    documents = list(loader.lazy_load())

    assert len(documents) == 1
    assert documents[0].metadata["url"] == "https://example.com/layer_url"
    # Add more assertions based on your expected behavior


def test_initialization_with_string_layer(  # type: ignore
    arcgis_mocks, mock_feature_layer, mock_gis
):
    layer_url = "https://example.com/layer_url"

    with patch("arcgis.features.FeatureLayer", return_value=mock_feature_layer):
        loader = ArcGISLoader(layer=layer_url, gis=mock_gis)

    assert loader.url == layer_url


def test_layer_description_provided_by_user(  # type: ignore
    arcgis_mocks, mock_feature_layer, mock_gis
):
    custom_description = "Custom Layer Description"
    loader = ArcGISLoader(
        layer=mock_feature_layer, gis=mock_gis, lyr_desc=custom_description
    )

    layer_properties = loader._get_layer_properties(lyr_desc=custom_description)

    assert layer_properties["layer_description"] == custom_description


def test_initialization_without_arcgis(mock_feature_layer, mock_gis):  # type: ignore
    with patch.dict("sys.modules", {"arcgis": None}):
        with pytest.raises(
            ImportError, match="arcgis is required to use the ArcGIS Loader"
        ):
            ArcGISLoader(layer=mock_feature_layer, gis=mock_gis)


def test_get_layer_properties_with_description(  # type: ignore
    arcgis_mocks, mock_feature_layer, mock_gis
):
    loader = ArcGISLoader(
        layer=mock_feature_layer, gis=mock_gis, lyr_desc="Custom Description"
    )

    props = loader._get_layer_properties("Custom Description")

    assert props["layer_description"] == "Custom Description"


def test_load_method(arcgis_mocks, mock_feature_layer, mock_gis):  # type: ignore
    loader = ArcGISLoader(layer=mock_feature_layer, gis=mock_gis)

    documents = loader.load()

    assert len(documents) == 1


def test_geometry_returned(arcgis_mocks, mock_feature_layer, mock_gis):  # type: ignore
    mock_feature_layer.query.return_value = [
        MagicMock(
            as_dict={
                "attributes": {"field": "value"},
                "geometry": {"type": "point", "coordinates": [0, 0]},
            }
        )
    ]

    loader = ArcGISLoader(layer=mock_feature_layer, gis=mock_gis, return_geometry=True)

    documents = list(loader.lazy_load())
    assert "geometry" in documents[0].metadata


def test_geometry_not_returned(  # type: ignore
    arcgis_mocks, mock_feature_layer, mock_gis
):
    loader = ArcGISLoader(layer=mock_feature_layer, gis=mock_gis, return_geometry=False)

    documents = list(loader.lazy_load())
    assert "geometry" not in documents[0].metadata
