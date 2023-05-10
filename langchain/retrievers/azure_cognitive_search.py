from __future__ import annotations

import json
from typing import Dict, List, Optional

import aiohttp
import requests
from pydantic import BaseModel, Extra, root_validator

from langchain.schema import BaseRetriever, Document
from langchain.utils import get_from_dict_or_env


class AzureCognitiveSearchRetriever(BaseRetriever, BaseModel):
    """Wrapper around Azure Cognitive Search."""

    service_name: str
    """Name of Azure Cognitive Search service"""
    index_name: str
    """Name of Index inside Azure Cognitive Search service"""
    api_key: str
    """API Key. Both Admin and Query keys work, but for reading data it's
    recommended to use a Query key."""
    api_version: str = "2020-06-30"
    """API version"""
    aiosession: Optional[aiohttp.ClientSession] = None
    """ClientSession, in case we want to reuse connection for better performance."""

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

    def _build_search_url(self) -> str:
        base_url = f"https://{self.service_name}.search.windows.net/"
        endpoint_path = f"indexes/{self.index_name}/docs?api-version={self.api_version}"
        return base_url + endpoint_path

    def _search(self, query: str) -> List[dict]:
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }
        search_url = f"{self._build_search_url()}&search={query}"
        response = requests.get(search_url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Error in search request: {response}")

        return json.loads(response.text)["value"]

    async def _async_search(self, query: str) -> List[dict]:
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }
        search_url = f"{self._build_search_url()}&search={query}"

        if not self.aiosession:
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, headers=headers) as response:
                    response_json = await response.json()
        else:
            async with self.aiosession.get(search_url, headers=headers) as response:
                response_json = await response.json()

        return response_json["value"]

    def get_relevant_documents(self, query: str) -> List[Document]:
        search_results = self._search(query)

        return [
            Document(page_content=result.pop("content"), metadata=result)
            for result in search_results
        ]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        search_results = await self._async_search(query)

        return [
            Document(page_content=result.pop("content"), metadata=result)
            for result in search_results
        ]
