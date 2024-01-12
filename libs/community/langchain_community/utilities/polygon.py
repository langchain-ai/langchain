"""
Util that calls several of Polygon's stock market REST APIs.
Docs: https://polygon.io/docs/stocks/getting-started
"""
import json
from typing import Dict, Optional

import requests
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.utils import get_from_dict_or_env

POLYGON_BASE_URL = "https://api.polygon.io/"


class PolygonAPIWrapper(BaseModel):
    """Wrapper for Polygon API."""

    polygon_api_key: Optional[str] = None

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key in environment."""
        polygon_api_key = get_from_dict_or_env(
            values, "polygon_api_key", "POLYGON_API_KEY"
        )
        values["polygon_api_key"] = polygon_api_key

        return values

    def get_last_quote(self, ticker: str) -> Optional[dict]:
        """Get the most recent National Best Bid and Offer (Quote) for a ticker."""
        url = f"{POLYGON_BASE_URL}v2/last/nbbo/{ticker}?apiKey={self.polygon_api_key}"
        response = requests.get(url)
        data = response.json()

        status = data.get("status", None)
        if status != "OK":
            raise ValueError(f"API Error: {data}")

        return data.get("results", None)

    def run(self, mode: str, ticker: str) -> str:
        if mode == "get_last_quote":
            return json.dumps(self.get_last_quote(ticker))
        else:
            raise ValueError(f"Invalid mode {mode} for Polygon API.")
