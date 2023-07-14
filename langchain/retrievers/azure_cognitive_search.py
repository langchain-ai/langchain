"""Retriever wrapper for Azure Cognitive Search."""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import aiohttp
import requests
from pydantic import Extra, root_validator

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.schema import BaseRetriever, Document
from langchain.utils import get_from_dict_or_env


class AzureCognitiveSearchRetriever(BaseRetriever):
    """Wrapper around Azure Cognitive Search."""

    service_name: str = ""
    """Name of Azure Cognitive Search service"""
    index_name: str = ""
    """Name of Index inside Azure Cognitive Search service"""
    api_key: str = ""
    """API Key. Both Admin and Query keys work, but for reading data it's
    recommended to use a Query key."""
    api_version: str = "2020-06-30"
    """API version"""
    aiosession: Optional[aiohttp.ClientSession] = None
    """ClientSession, in case we want to reuse connection for better performance."""
    content_key: str = "content"
    """Key in a retrieved result to set as the Document page_content."""
    top_k: int = 3
    """No of results to return"""
    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that service name, index name and api key exists in environment."""
        values["service_name"] = get_from_dict_or_env(
            values, "service_name", "AZURE_COGNITIVE_SEARCH_SERVICE_NAME"
        )
        values["index_name"] = get_from_dict_or_env(
            values, "index_name", "AZURE_COGNITIVE_SEARCH_INDEX_NAME"
        )
        values["api_key"] = get_from_dict_or_env(
            values, "api_key", "AZURE_COGNITIVE_SEARCH_API_KEY"
        )
        return values


    def _build_search_url(self, query: str, top_k: int) -> str:
        """Builds the search URL for querying the search service.
        Args:
            query (str): The search query to be executed.
            top_k (int): The number of search results to retrieve.
        Returns:
            str: The constructed search URL.
        Raises:
            None
        Examples:
            >>> client = SearchClient()
            >>> url = client._build_search_url("example query", 10)
            >>> print(url)
            https://search-service.search.windows.net/indexes/index-name/docs?api-version=api-version&search=example%20query&top=10
        """
        base_url = f"https://{self.service_name}.search.windows.net/"
        endpoint_path = f"indexes/{self.index_name}/docs?api-version={self.api_version}"
        return base_url + endpoint_path + f"&search={query}" + f"&top={top_k}"

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }

    def _search(self, query: str, top_k: int) -> List[dict]:
        search_url = self._build_search_url(query, top_k)
        response = requests.get(search_url, headers=self._headers)
        if response.status_code != 200:
            raise Exception(f"Error in search request: {response}")

        return json.loads(response.text)["value"]

    async def _asearch(self, query: str, top_k: int) -> List[dict]:
        search_url = self._build_search_url(query, top_k)
        if not self.aiosession:
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, headers=self._headers) as response:
                    response_json = await response.json()
        else:
            async with self.aiosession.get(
                search_url, headers=self._headers
            ) as response:
                response_json = await response.json()

        return response_json["value"]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        search_results = self._search(query, self.top_k)

        return [
            Document(page_content=result.pop(self.content_key), metadata=result)
            for result in search_results
        ]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        search_results = await self._asearch(query, self.top_k)

        return [
            Document(page_content=result.pop(self.content_key), metadata=result)
            for result in search_results
        ]
