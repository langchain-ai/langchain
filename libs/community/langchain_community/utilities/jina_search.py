import json
from typing import Any, Dict, List

import requests
from langchain_core.documents import Document
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, SecretStr, model_validator
from yarl import URL


class JinaSearchAPIWrapper(BaseModel):
    """Wrapper around the Jina search engine."""

    api_key: SecretStr

    base_url: str = "https://s.jina.ai/"
    """The base URL for the Jina search engine."""

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and endpoint exists in environment."""
        api_key = get_from_dict_or_env(values, "api_key", "JINA_API_KEY")
        values["api_key"] = api_key

        return values

    def run(self, query: str) -> str:
        """Query the Jina search engine and return the results as a JSON string.

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
                "content": item.get("content"),
            }
            for item in web_search_results
        ]
        return json.dumps(final_results)

    def download_documents(self, query: str) -> List[Document]:
        """Query the Jina search engine and return the results as a list of Documents.

        Args:
            query: The query to search for.

        Returns: The results as a list of Documents.

        """
        results = self._search_request(query)
        return [
            Document(
                page_content=item.get("content"),  # type: ignore[arg-type]
                metadata={
                    "title": item.get("title"),
                    "link": item.get("url"),
                    "description": item.get("description"),
                },
            )
            for item in results
        ]

    def _search_request(self, query: str) -> List[dict]:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
        }
        url = str(URL(self.base_url + query))
        response = requests.get(url, headers=headers)
        if not response.ok:
            raise Exception(f"HTTP error {response.status_code}")

        return response.json().get("data", [])
