import json
from typing import List

import requests
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel
from yarl import URL


class JinaSearchAPIWrapper(BaseModel):
    """Wrapper around the Jina search engine."""

    base_url: str = "https://s.jina.ai/"
    """The base URL for the Jina search engine."""

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
        }
        url = str(URL(self.base_url + query))
        response = requests.get(url, headers=headers)
        if not response.ok:
            raise Exception(f"HTTP error {response.status_code}")

        return response.json().get("data", [])
