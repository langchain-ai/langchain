"""
Util that calls several of Polygon's stock market REST APIs.
Docs: https://polygon.io/docs/stocks/getting-started
"""

import json
from typing import Any, Dict, Optional

import requests
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, model_validator

POLYGON_BASE_URL = "https://api.polygon.io/"


class PolygonAPIWrapper(BaseModel):
    """Wrapper for Polygon API."""

    polygon_api_key: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key in environment."""
        polygon_api_key = get_from_dict_or_env(
            values, "polygon_api_key", "POLYGON_API_KEY"
        )
        values["polygon_api_key"] = polygon_api_key

        return values

    def get_financials(self, ticker: str) -> Optional[dict]:
        """
        Get fundamental financial data, which is found in balance sheets,
        income statements, and cash flow statements for a given ticker.

        /vX/reference/financials
        """
        url = (
            f"{POLYGON_BASE_URL}vX/reference/financials?"
            f"ticker={ticker}&"
            f"apiKey={self.polygon_api_key}"
        )
        response = requests.get(url)
        data = response.json()

        status = data.get("status", None)
        if status not in ("OK", "STOCKBUSINESS", "STOCKSBUSINESS"):
            raise ValueError(f"API Error: {data}")

        return data.get("results", None)

    def get_last_quote(self, ticker: str) -> Optional[dict]:
        """
        Get the most recent National Best Bid and Offer (Quote) for a ticker.

        /v2/last/nbbo/{ticker}
        """
        url = f"{POLYGON_BASE_URL}v2/last/nbbo/{ticker}?apiKey={self.polygon_api_key}"
        response = requests.get(url)
        data = response.json()

        status = data.get("status", None)
        if status not in ("OK", "STOCKBUSINESS", "STOCKSBUSINESS"):
            raise ValueError(f"API Error: {data}")

        return data.get("results", None)

    def get_ticker_news(self, ticker: str) -> Optional[dict]:
        """
        Get the most recent news articles relating to a stock ticker symbol,
        including a summary of the article and a link to the original source.

        /v2/reference/news
        """
        url = (
            f"{POLYGON_BASE_URL}v2/reference/news?"
            f"ticker={ticker}&"
            f"apiKey={self.polygon_api_key}"
        )
        response = requests.get(url)
        data = response.json()

        status = data.get("status", None)
        if status not in ("OK", "STOCKBUSINESS", "STOCKSBUSINESS"):
            raise ValueError(f"API Error: {data}")

        return data.get("results", None)

    def get_aggregates(self, ticker: str, **kwargs: Any) -> Optional[dict]:
        """
        Get aggregate bars for a stock over a given date range
        in custom time window sizes.

        /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}
        """
        timespan = kwargs.get("timespan", "day")
        multiplier = kwargs.get("timespan_multiplier", 1)
        from_date = kwargs.get("from_date", None)
        to_date = kwargs.get("to_date", None)
        adjusted = kwargs.get("adjusted", True)
        sort = kwargs.get("sort", "asc")

        url = (
            f"{POLYGON_BASE_URL}v2/aggs"
            f"/ticker/{ticker}"
            f"/range/{multiplier}"
            f"/{timespan}"
            f"/{from_date}"
            f"/{to_date}"
            f"?apiKey={self.polygon_api_key}"
            f"&adjusted={adjusted}"
            f"&sort={sort}"
        )
        response = requests.get(url)
        data = response.json()

        status = data.get("status", None)
        if status not in ("OK", "STOCKBUSINESS", "STOCKSBUSINESS"):
            raise ValueError(f"API Error: {data}")

        return data.get("results", None)

    def run(self, mode: str, ticker: str, **kwargs: Any) -> str:
        if mode == "get_financials":
            return json.dumps(self.get_financials(ticker))
        elif mode == "get_last_quote":
            return json.dumps(self.get_last_quote(ticker))
        elif mode == "get_ticker_news":
            return json.dumps(self.get_ticker_news(ticker))
        elif mode == "get_aggregates":
            return json.dumps(self.get_aggregates(ticker, **kwargs))
        else:
            raise ValueError(f"Invalid mode {mode} for Polygon API.")
