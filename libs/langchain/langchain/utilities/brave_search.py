import json
from typing import List

import requests

from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema import Document


class BraveSearchWrapper(BaseModel):
    """Wrapper around the Brave search engine."""

    api_key: str
    """The API key to use for the Brave search engine."""
    search_kwargs: dict = Field(default_factory=dict)
    """Additional keyword arguments to pass to the search request."""
    base_url: str = "https://api.search.brave.com/res/v1/web/search"
    """The base URL for the Brave search engine."""

    def run(self, query: str) -> str:
        """Query the Brave search engine and return the results as a JSON string.

        Args:
            query: The query to search for.

        Returns: The results as a JSON string.

        """
        web_search_results = self._search_request(query=query)
        final_results = [
            {
                "title": item.get("title"),
                "link": item.get("url"),
                "snippet": item.get("description"),
            }
            for item in web_search_results
        ]
        return json.dumps(final_results)

    def download_documents(self, query: str) -> List[Document]:
        """Query the Brave search engine and return the results as a list of Documents.

        Args:
            query: The query to search for.

        Returns: The results as a list of Documents.

        """
        results = self._search_request(query)
        return [
            Document(
                page_content=item.get("description"),
                metadata={"title": item.get("title"), "link": item.get("url")},
            )
            for item in results
        ]

    def _search_request(self, query: str) -> List[dict]:
        headers = {
            "X-Subscription-Token": self.api_key,
            "Accept": "application/json",
        }
        req = requests.PreparedRequest()
        params = {**self.search_kwargs, **{"q": query}}
        req.prepare_url(self.base_url, params)
        if req.url is None:
            raise ValueError("prepared url is None, this should not happen")

        response = requests.get(req.url, headers=headers)
        if not response.ok:
            raise Exception(f"HTTP error {response.status_code}")

        return response.json().get("web", {}).get("results", [])
