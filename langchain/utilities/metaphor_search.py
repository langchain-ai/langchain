"""Util that calls Metaphor Search API.

In order to set this up, follow instructions at:
"""
import json
from typing import Dict, List

import aiohttp
import requests
from pydantic import BaseModel, Extra, root_validator

from langchain.utils import get_from_dict_or_env

METAPHOR_API_URL = "https://api.metaphor.systems"


class MetaphorSearchAPIWrapper(BaseModel):
    """Wrapper for Metaphor Search API."""

    metaphor_api_key: str
    k: int = 10

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def _metaphor_search_results(self, query: str, num_results: int) -> List[dict]:
        headers = {"X-Api-Key": self.metaphor_api_key}
        params = {"numResults": num_results, "query": query}
        response = requests.post(
            # type: ignore
            f"{METAPHOR_API_URL}/search",
            headers=headers,
            json=params,
        )

        response.raise_for_status()
        search_results = response.json()
        print(search_results)
        return search_results["results"]

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and endpoint exists in environment."""
        metaphor_api_key = get_from_dict_or_env(
            values, "metaphor_api_key", "METAPHOR_API_KEY"
        )
        values["metaphor_api_key"] = metaphor_api_key

        return values

    def results(self, query: str, num_results: int) -> List[Dict]:
        """Run query through Metaphor Search and return metadata.

        Args:
            query: The query to search for.
            num_results: The number of results to return.

        Returns:
            A list of dictionaries with the following keys:
                title - The title of the
                url - The url
                author - Author of the content, if applicable. Otherwise, None.
                date_created - Estimated date created,
                    in YYYY-MM-DD format. Otherwise, None.
        """
        raw_search_results = self._metaphor_search_results(
            query, num_results=num_results
        )
        return self._clean_results(raw_search_results)

    async def results_async(self, query: str, num_results: int) -> List[Dict]:
        """Get results from the Metaphor Search API asynchronously."""

        # Function to perform the API call
        async def fetch() -> str:
            headers = {"X-Api-Key": self.metaphor_api_key}
            params = {"numResults": num_results, "query": query}
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{METAPHOR_API_URL}/search", json=params, headers=headers
                ) as res:
                    if res.status == 200:
                        data = await res.text()
                        return data
                    else:
                        raise Exception(f"Error {res.status}: {res.reason}")

        results_json_str = await fetch()
        results_json = json.loads(results_json_str)
        return self._clean_results(results_json["results"])

    def _clean_results(self, raw_search_results: List[Dict]) -> List[Dict]:
        cleaned_results = []
        for result in raw_search_results:
            cleaned_results.append(
                {
                    "title": result["title"],
                    "url": result["url"],
                    "author": result["author"],
                    "date_created": result["dateCreated"],
                }
            )
        return cleaned_results
