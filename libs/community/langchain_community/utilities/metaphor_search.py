"""Util that calls Metaphor Search API.

In order to set this up, follow instructions at:
"""

import json
from typing import Any, Dict, List, Optional

import aiohttp
import requests
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, model_validator

METAPHOR_API_URL = "https://api.metaphor.systems"


class MetaphorSearchAPIWrapper(BaseModel):
    """Wrapper for Metaphor Search API."""

    metaphor_api_key: str
    k: int = 10

    model_config = ConfigDict(
        extra="forbid",
    )

    def _metaphor_search_results(
        self,
        query: str,
        num_results: int,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        use_autoprompt: Optional[bool] = None,
    ) -> List[dict]:
        headers = {"X-Api-Key": self.metaphor_api_key}
        params = {
            "numResults": num_results,
            "query": query,
            "includeDomains": include_domains,
            "excludeDomains": exclude_domains,
            "startCrawlDate": start_crawl_date,
            "endCrawlDate": end_crawl_date,
            "startPublishedDate": start_published_date,
            "endPublishedDate": end_published_date,
            "useAutoprompt": use_autoprompt,
        }
        response = requests.post(
            f"{METAPHOR_API_URL}/search",
            headers=headers,
            json=params,
        )

        response.raise_for_status()
        search_results = response.json()
        return search_results["results"]

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and endpoint exists in environment."""
        metaphor_api_key = get_from_dict_or_env(
            values, "metaphor_api_key", "METAPHOR_API_KEY"
        )
        values["metaphor_api_key"] = metaphor_api_key

        return values

    def results(
        self,
        query: str,
        num_results: int,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        use_autoprompt: Optional[bool] = None,
    ) -> List[Dict]:
        """Run query through Metaphor Search and return metadata.

        Args:
            query: The query to search for.
            num_results: The number of results to return.
            include_domains: A list of domains to include in the search. Only one of include_domains and exclude_domains should be defined.
            exclude_domains: A list of domains to exclude from the search. Only one of include_domains and exclude_domains should be defined.
            start_crawl_date: If specified, only pages we crawled after start_crawl_date will be returned.
            end_crawl_date: If specified, only pages we crawled before end_crawl_date will be returned.
            start_published_date: If specified, only pages published after start_published_date will be returned.
            end_published_date: If specified, only pages published before end_published_date will be returned.
            use_autoprompt: If true, we turn your query into a more Metaphor-friendly query. Adds latency.

        Returns:
            A list of dictionaries with the following keys:
                title - The title of the page
                url - The url
                author - Author of the content, if applicable. Otherwise, None.
                published_date - Estimated date published
                    in YYYY-MM-DD format. Otherwise, None.
        """  # noqa: E501
        raw_search_results = self._metaphor_search_results(
            query,
            num_results=num_results,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            start_crawl_date=start_crawl_date,
            end_crawl_date=end_crawl_date,
            start_published_date=start_published_date,
            end_published_date=end_published_date,
            use_autoprompt=use_autoprompt,
        )
        return self._clean_results(raw_search_results)

    async def results_async(
        self,
        query: str,
        num_results: int,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        use_autoprompt: Optional[bool] = None,
    ) -> List[Dict]:
        """Get results from the Metaphor Search API asynchronously."""

        # Function to perform the API call
        async def fetch() -> str:
            headers = {"X-Api-Key": self.metaphor_api_key}
            params = {
                "numResults": num_results,
                "query": query,
                "includeDomains": include_domains,
                "excludeDomains": exclude_domains,
                "startCrawlDate": start_crawl_date,
                "endCrawlDate": end_crawl_date,
                "startPublishedDate": start_published_date,
                "endPublishedDate": end_published_date,
                "useAutoprompt": use_autoprompt,
            }
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
                    "title": result.get("title", "Unknown Title"),
                    "url": result.get("url", "Unknown URL"),
                    "author": result.get("author", "Unknown Author"),
                    "published_date": result.get("publishedDate", "Unknown Date"),
                }
            )
        return cleaned_results
