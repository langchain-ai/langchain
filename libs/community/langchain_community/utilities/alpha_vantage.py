"""Util that calls AlphaVantage for Currency Exchange Rate."""
from typing import Any, Dict, List, Optional

import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env


class AlphaVantageAPIWrapper(BaseModel):
    """Wrapper for AlphaVantage API for Currency Exchange Rate.

    Docs for using:

    1. Go to AlphaVantage and sign up for an API key
    2. Save your API KEY into ALPHAVANTAGE_API_KEY env variable
    """

    alphavantage_api_key: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        values["alphavantage_api_key"] = get_from_dict_or_env(
            values, "alphavantage_api_key", "ALPHAVANTAGE_API_KEY"
        )
        return values

    def search_symbols(self, keywords: str) -> Dict[str, Any]:
        """Make a request to the AlphaVantage API to search for symbols."""
        response = requests.get(
            "https://www.alphavantage.co/query/",
            params={
                "function": "SYMBOL_SEARCH",
                "keywords": keywords,
                "apikey": self.alphavantage_api_key,
            },
        )
        response.raise_for_status()
        data = response.json()

        if "Error Message" in data:
            raise ValueError(f"API Error: {data['Error Message']}")

        return data

    def _get_market_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Make a request to the AlphaVantage API to get market news sentiment for a
        given symbol."""
        response = requests.get(
            "https://www.alphavantage.co/query/",
            params={
                "function": "NEWS_SENTIMENT",
                "symbol": symbol,
                "apikey": self.alphavantage_api_key,
            },
        )
        response.raise_for_status()
        data = response.json()

        if "Error Message" in data:
            raise ValueError(f"API Error: {data['Error Message']}")

        return data

    def _get_time_series_daily(self, symbol: str) -> Dict[str, Any]:
        """Make a request to the AlphaVantage API to get the daily time series."""
        response = requests.get(
            "https://www.alphavantage.co/query/",
            params={
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "apikey": self.alphavantage_api_key,
            },
        )
        response.raise_for_status()
        data = response.json()

        if "Error Message" in data:
            raise ValueError(f"API Error: {data['Error Message']}")

        return data

    def _get_quote_endpoint(self, symbol: str) -> Dict[str, Any]:
        """Make a request to the AlphaVantage API to get the
        latest price and volume information."""
        response = requests.get(
            "https://www.alphavantage.co/query/",
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": self.alphavantage_api_key,
            },
        )
        response.raise_for_status()
        data = response.json()

        if "Error Message" in data:
            raise ValueError(f"API Error: {data['Error Message']}")

        return data

    def _get_time_series_weekly(self, symbol: str) -> Dict[str, Any]:
        """Make a request to the AlphaVantage API
        to get the Weekly Time Series."""
        response = requests.get(
            "https://www.alphavantage.co/query/",
            params={
                "function": "TIME_SERIES_WEEKLY",
                "symbol": symbol,
                "apikey": self.alphavantage_api_key,
            },
        )
        response.raise_for_status()
        data = response.json()

        if "Error Message" in data:
            raise ValueError(f"API Error: {data['Error Message']}")

        return data

    def _get_top_gainers_losers(self) -> Dict[str, Any]:
        """Make a request to the AlphaVantage API to get the top gainers, losers,
        and most actively traded tickers in the US market."""
        response = requests.get(
            "https://www.alphavantage.co/query/",
            params={
                "function": "TOP_GAINERS_LOSERS",
                "apikey": self.alphavantage_api_key,
            },
        )
        response.raise_for_status()
        data = response.json()

        if "Error Message" in data:
            raise ValueError(f"API Error: {data['Error Message']}")

        return data

    def _get_exchange_rate(
        self, from_currency: str, to_currency: str
    ) -> Dict[str, Any]:
        """Make a request to the AlphaVantage API to get the exchange rate."""
        response = requests.get(
            "https://www.alphavantage.co/query/",
            params={
                "function": "CURRENCY_EXCHANGE_RATE",
                "from_currency": from_currency,
                "to_currency": to_currency,
                "apikey": self.alphavantage_api_key,
            },
        )
        response.raise_for_status()
        data = response.json()

        if "Error Message" in data:
            raise ValueError(f"API Error: {data['Error Message']}")

        return data

    @property
    def standard_currencies(self) -> List[str]:
        return ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]

    def run(self, from_currency: str, to_currency: str) -> str:
        """Get the current exchange rate for a specified currency pair."""
        if to_currency not in self.standard_currencies:
            from_currency, to_currency = to_currency, from_currency

        data = self._get_exchange_rate(from_currency, to_currency)
        return data["Realtime Currency Exchange Rate"]
