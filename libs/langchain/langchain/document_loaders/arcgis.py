"""Document Loader for ArcGIS FeatureLayers."""

import json
import re
import warnings
from typing import TYPE_CHECKING, Optional, List, Union, Iterator

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


if TYPE_CHECKING:
    import arcgis  # type: ignore


class ArcGISLoader(BaseLoader):
    """Load records from an ArcGIS FeatureLayer."""

    def _get_layer_properties(self) -> dict:
        """Get the layer properties from the FeatureLayer."""

        def extract_text(possibly_html: str) -> str:
            soup = self.BEAUTIFULSOUP(possibly_html, features="lxml")
            text = soup.text
            return text

        layer_number_pattern = re.compile(r"/\d+$")
        not_provided = "(Not Provided)"
        props = self.layer.properties

        try:
            if self.BEAUTIFULSOUP:
                lyr_desc = extract_text(props["description"]) or not_provided
            else:
                lyr_desc = props["description"] or not_provided
        except KeyError:
            lyr_desc = not_provided
        try:
            item_id = props["serviceItemId"]
            item = self.gis.content.get(item_id) or arcgis.features.FeatureLayer(
                re.sub(layer_number_pattern, "", self.url),
            )
            try:
                raw_desc = item.description
            except AttributeError:
                raw_desc = item.properties.description
            if self.BEAUTIFULSOUP:
                item_desc = extract_text(raw_desc) or not_provided
            else:
                item_desc = raw_desc or not_provided
        except KeyError:
            item_desc = not_provided
        return {
            "layer_description": lyr_desc,
            "item_description": item_desc,
            "layer_properties": props,
        }

    def __init__(
        self,
        layer: Union[str, arcgis.features.FeatureLayer],
        gis: Optional[arcgis.gis.GIS] = None,
        where: str = "1=1",
        out_fields: Optional[Union[List[str], str]] = None,
        return_geometry: bool = False,
        **kwargs,
    ):
        try:
            import arcgis
        except ImportError as e:
            raise ImportError(
                "arcgis is required to use the ArcGIS Loader. "
                "Install it with pip or conda."
            ) from e

        try:
            from bs4 import BeautifulSoup  # type: ignore

            self.BEAUTIFULSOUP = BeautifulSoup
        except ImportError:
            warnings.warn("BeautifulSoup not found. HTML will not be parsed.")
            self.BEAUTIFULSOUP = None

        self.gis = gis or arcgis.gis.GIS()

        if isinstance(layer, str):
            self.url = layer
            self.layer = arcgis.features.FeatureLayer(layer, gis=gis)
        else:
            self.url = layer.url
            self.layer = layer

        self.layer_properties = self._get_layer_properties()

        self.where = where

        if isinstance(out_fields, str):
            self.out_fields = out_fields
        elif out_fields is None:
            self.out_fields = "*"
        else:
            self.out_fields = ",".join(out_fields)

        self.return_geometry = return_geometry
        self.kwargs = kwargs

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load records from FeatureLayer."""

        query_response = self.layer.query(
            where=self.where,
            out_fields=self.out_fields,
            return_geometry=self.return_geometry,
            return_all_records=True,
            **self.kwargs,
        )
        features = (feature.as_dict["attributes"] for feature in query_response)
        for feature in features:
            yield Document(
                page_content=json.dumps(feature),
                metadata={
                    "url": self.url,
                    "layer_description": self.layer_properties["layer_description"],
                    "item_description": self.layer_properties["item_description"],
                    "layer_properties": self.layer_properties["layer_properties"],
                },
            )

    def load(self) -> List[Document]:
        """Load all records from FeatureLayer."""
        return list(self.lazy_load())
