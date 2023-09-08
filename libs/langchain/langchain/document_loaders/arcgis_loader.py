"""Document Loader for ArcGIS FeatureLayers."""

from __future__ import annotations

import json
import re
import warnings
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Union

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

if TYPE_CHECKING:
    import arcgis

_NOT_PROVIDED = "(Not Provided)"


class ArcGISLoader(BaseLoader):
    """Load records from an ArcGIS FeatureLayer."""

    def __init__(
        self,
        layer: Union[str, arcgis.features.FeatureLayer],
        gis: Optional[arcgis.gis.GIS] = None,
        where: str = "1=1",
        out_fields: Optional[Union[List[str], str]] = None,
        return_geometry: bool = False,
        return_all_records: bool = True,
        **kwargs: Any,
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
        self.return_all_records = return_all_records
        self.kwargs = kwargs

    def _get_layer_properties(self) -> dict:
        """Get the layer properties from the FeatureLayer."""
        import arcgis

        layer_number_pattern = re.compile(r"/\d+$")
        props = self.layer.properties

        try:
            if self.BEAUTIFULSOUP:
                lyr_desc = self.BEAUTIFULSOUP(props["description"]).text
            else:
                lyr_desc = props["description"]
            lyr_desc = lyr_desc or _NOT_PROVIDED
        except KeyError:
            lyr_desc = _NOT_PROVIDED
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
                item_desc = self.BEAUTIFULSOUP(raw_desc).text
            else:
                item_desc = raw_desc
            item_desc = item_desc or _NOT_PROVIDED
        except KeyError:
            item_desc = _NOT_PROVIDED
        return {
            "layer_description": lyr_desc,
            "item_description": item_desc,
            "layer_properties": props,
        }

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load records from FeatureLayer."""

        query_response = self.layer.query(
            where=self.where,
            out_fields=self.out_fields,
            return_geometry=self.return_geometry,
            return_all_records=self.return_all_records,
            **self.kwargs,
        )
        features = (feature.as_dict["attributes"] for feature in query_response)
        for feature in features:
            yield Document(
                page_content=json.dumps(feature),
                metadata={
                    "accessed": f"{datetime.now(timezone.utc).isoformat()}Z",
                    "name": self.layer_properties["layer_properties"]["name"],
                    "url": self.url,
                    "layer_description": self.layer_properties["layer_description"],
                    "item_description": self.layer_properties["item_description"],
                    "layer_properties": self.layer_properties["layer_properties"],
                },
            )

    def load(self) -> List[Document]:
        """Load all records from FeatureLayer."""
        return list(self.lazy_load())
