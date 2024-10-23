from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterator, Optional

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.utilities import OpenSkyAPIWrapper


class OpenSkyLoader(BaseLoader):
    """Load flight data from the OpenSky API."""

    def __init__(
        self,
        api_wrapper: OpenSkyAPIWrapper,
        mode: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the loader with the API wrapper and parameters."""
        super().__init__()
        self.api_wrapper = api_wrapper
        self.mode = mode
        self.params = params or {}

    def lazy_load(self) -> Iterator[Document]:
        """Lazily load flight data based on the specified flight type."""
        if self.mode == "all_states":
            data = self.api_wrapper.fetch_all_state_vectors()
        elif self.mode == "by_interval":
            data = self.api_wrapper.fetch_flights_by_time_interval(**self.params)
        elif self.mode == "by_aircraft":
            data = self.api_wrapper.fetch_flights_by_aircraft(**self.params)
        elif self.mode == "arrivals":
            data = self.api_wrapper.fetch_arrivals_by_airport(**self.params)
        elif self.mode == "departures":
            data = self.api_wrapper.fetch_departures_by_airport(**self.params)
        else:
            raise ValueError("Invalid flight type specified.")

        # Create Document objects from the fetched data
        for flight in data:
            metadata = {
                "mode": self.mode,
                "queried_at": datetime.now(),
            }
            yield Document(page_content=str(flight), metadata=metadata)
