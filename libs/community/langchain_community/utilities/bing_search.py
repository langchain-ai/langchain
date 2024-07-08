"""Util that calls Bing Search."""

from typing import Dict, List

import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env

# BING_SEARCH_ENDPOINT is the default endpoint for Bing Web Search API.
# Currently There are two web-based Bing Search services available on Azure,
# i.e. Bing Web Search[1] and Bing Custom Search[2]. Compared to Bing Custom Search,
# Both services that provides a wide range of search results, while Bing Custom
# Search requires you to provide an additional custom search instance, `customConfig`.
# Both services are available for BingSearchAPIWrapper.
# History of Azure Bing Search API:
# Before shown in Azure Marketplace as a separate service, Bing Search APIs were
# part of Azure Cognitive Services, the endpoint of which is unique, and the user
# must specify the endpoint when making a request. After transitioning to Azure
# Marketplace, the endpoint is standardized and the user does not need to specify
# the endpoint[3].
# Reference:
#  1. https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/overview
#  2. https://learn.microsoft.com/en-us/bing/search-apis/bing-custom-search/overview
#  3. https://azure.microsoft.com/en-in/updates/bing-search-apis-will-transition-from-azure-cognitive-services-to-azure-marketplace-on-31-october-2023/
DEFAULT_BING_SEARCH_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"


class BingSearchAPIWrapper(BaseModel):
    """Wrapper for Bing Web Search API."""

    bing_subscription_key: str
    bing_search_url: str
    k: int = 10
    search_kwargs: dict = Field(default_factory=dict)
    """Additional keyword arguments to pass to the search request."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def _bing_search_results(self, search_term: str, count: int) -> List[dict]:
        headers = {"Ocp-Apim-Subscription-Key": self.bing_subscription_key}
        params = {
            "q": search_term,
            "count": count,
            "textDecorations": True,
            "textFormat": "HTML",
            **self.search_kwargs,
        }
        response = requests.get(
            self.bing_search_url,
            headers=headers,
            params=params,  # type: ignore
        )
        response.raise_for_status()
        search_results = response.json()
        if "webPages" in search_results:
            return search_results["webPages"]["value"]
        return []

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and endpoint exists in environment."""
        bing_subscription_key = get_from_dict_or_env(
            values, "bing_subscription_key", "BING_SUBSCRIPTION_KEY"
        )
        values["bing_subscription_key"] = bing_subscription_key

        bing_search_url = get_from_dict_or_env(
            values,
            "bing_search_url",
            "BING_SEARCH_URL",
            default=DEFAULT_BING_SEARCH_ENDPOINT,
        )

        values["bing_search_url"] = bing_search_url

        return values

    def run(self, query: str) -> str:
        """Run query through BingSearch and parse result."""
        snippets = []
        results = self._bing_search_results(query, count=self.k)
        if len(results) == 0:
            return "No good Bing Search Result was found"
        for result in results:
            snippets.append(result["snippet"])

        return " ".join(snippets)

    def results(self, query: str, num_results: int) -> List[Dict]:
        """Run query through BingSearch and return metadata.

        Args:
            query: The query to search for.
            num_results: The number of results to return.

        Returns:
            A list of dictionaries with the following keys:
                snippet - The description of the result.
                title - The title of the result.
                link - The link to the result.
        """
        metadata_results = []
        results = self._bing_search_results(query, count=num_results)
        if len(results) == 0:
            return [{"Result": "No good Bing Search Result was found"}]
        for result in results:
            metadata_result = {
                "snippet": result["snippet"],
                "title": result["name"],
                "link": result["url"],
            }
            metadata_results.append(metadata_result)

        return metadata_results
