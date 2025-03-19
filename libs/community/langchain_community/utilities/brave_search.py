import json
from typing import Dict, List

import requests
from langchain_core.documents import Document
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, Field, SecretStr, model_validator


class BraveSearchWrapper(BaseModel):
    """Wrapper around the Brave search engine."""

    api_key: SecretStr
    """The API key to use for the Brave search engine."""
    search_kwargs: dict = Field(default_factory=dict)
    """Additional keyword arguments to pass to the search request."""
    base_url: str = "https://api.search.brave.com/res/v1/web/search"
    """The base URL for the Brave search engine."""

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""

        # if the api key is not in the values, get it from the environment, or we fail
        # this is the pattern used in other tools, like Tavily, but it's not ideal from
        # linter's point of view so we have to add some ignore comments
        api_key = get_from_dict_or_env(values, "api_key", "BRAVE_SEARCH_API_KEY")
        values["api_key"] = api_key

        return values

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
                "snippet": " ".join(
                    filter(
                        None, [item.get("description"), *item.get("extra_snippets", [])]
                    )
                ),
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
                page_content=" ".join(
                    filter(
                        None, [item.get("description"), *item.get("extra_snippets", [])]
                    )
                ),
                metadata={"title": item.get("title"), "link": item.get("url")},
            )
            for item in results
        ]

    def _search_request(self, query: str) -> List[dict]:
        headers = {
            "X-Subscription-Token": self.api_key.get_secret_value(),
            "Accept": "application/json",
        }
        req = requests.PreparedRequest()
        params = {**self.search_kwargs, **{"q": query, "extra_snippets": True}}
        req.prepare_url(self.base_url, params)
        if req.url is None:
            raise ValueError("prepared url is None, this should not happen")

        response = requests.get(req.url, headers=headers)
        if not response.ok:
            raise Exception(f"HTTP error {response.status_code}")

        return response.json().get("web", {}).get("results", [])
