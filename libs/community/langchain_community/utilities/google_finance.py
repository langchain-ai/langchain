"""Util that calls Google Finance Search."""
from typing import Any, Dict, Optional, cast

from langchain_core.pydantic_v1 import BaseModel, Extra, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env


class GoogleFinanceAPIWrapper(BaseModel):
    """Wrapper for SerpApi's Google Finance API

    You can create SerpApi.com key by signing up at: https://serpapi.com/users/sign_up.
    The wrapper uses the SerpApi.com python package:
    https://serpapi.com/integrations/python
    To use, you should have the environment variable ``SERPAPI_API_KEY``
    set with your API key, or pass `serp_api_key` as a named parameter
    to the constructor.
    Example:
        .. code-block:: python
        from langchain_community.utilities import GoogleFinanceAPIWrapper
        google_Finance = GoogleFinanceAPIWrapper()
        google_Finance.run('langchain')
    """

    serp_search_engine: Any
    serp_api_key: Optional[SecretStr] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["serp_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "serp_api_key", "SERPAPI_API_KEY")
        )

        try:
            from serpapi import SerpApiClient

        except ImportError:
            raise ImportError(
                "google-search-results is not installed. "
                "Please install it with `pip install google-search-results"
                ">=2.4.2`"
            )
        serp_search_engine = SerpApiClient
        values["serp_search_engine"] = serp_search_engine

        return values

    def run(self, query: str) -> str:
        """Run query through Google Finance with Serpapi"""
        serpapi_api_key = cast(SecretStr, self.serp_api_key)
        params = {
            "engine": "google_finance",
            "api_key": serpapi_api_key.get_secret_value(),
            "q": query,
        }

        total_results = {}
        client = self.serp_search_engine(params)
        total_results = client.get_dict()

        if not total_results:
            return "Nothing was found from the query: " + query

        markets = total_results.get("markets", {})
        res = "\nQuery: " + query + "\n"

        if "futures_chain" in total_results:
            futures_chain = total_results.get("futures_chain", [])[0]
            stock = futures_chain["stock"]
            price = futures_chain["price"]
            temp = futures_chain["price_movement"]
            percentage = temp["percentage"]
            movement = temp["movement"]
            res += (
                f"stock: {stock}\n"
                + f"price: {price}\n"
                + f"percentage: {percentage}\n"
                + f"movement: {movement}\n"
            )

        else:
            res += "No summary information\n"

        for key in markets:
            if (key == "us") or (key == "asia") or (key == "europe"):
                res += key
                res += ": price = "
                res += str(markets[key][0]["price"])
                res += ", movement = "
                res += markets[key][0]["price_movement"]["movement"]
                res += "\n"

        return res
