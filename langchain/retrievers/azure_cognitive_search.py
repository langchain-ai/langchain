from __future__ import annotations

from typing import List, Dict, Optional

import json
import aiohttp
import requests
from pydantic import BaseModel, Extra, root_validator

from langchain.schema import BaseRetriever, Document
from langchain.utils import get_from_dict_or_env


class AzureCognitiveSearchRetriever(BaseRetriever, BaseModel):
    service_name: str  # name of Azure Cognitive Search service
    index_name: str
    azure_cognitive_search_api_key: Optional[str] = None
    aiosession: Optional[aiohttp.ClientSession] = None

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        api_key = get_from_dict_or_env(
            values, "azure_cognitive_search_api_key", "AZURE_COGNITIVE_SEARCH_API_KEY"
        )
        values["azure_cognitive_search_api_key"] = api_key
        return values


    def _build_search_url(self, api_version: str = "2020-06-30") -> str:
        return f"https://{self.service_name}.search.windows.net/indexes/{self.index_name}/docs?api-version={api_version}"

    def _search(self, query: str) -> List[dict]:
        headers = {
            "Content-Type": "application/json",
            "api-key": self.azure_cognitive_search_api_key,
        }
        search_url = f"{self._build_search_url()}&search={query}"
        response = requests.get(search_url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Error in search request: {response}")

        return json.loads(response.text)["value"]

    async def _async_search(self, query: str) -> List[dict]:
        headers = {
            "Content-Type": "application/json",
            "api-key": self.azure_cognitive_search_api_key,
        }
        search_url = f"{self._build_search_url()}&search={query}"

        if not self.aiosession:
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, headers=headers) as response:
                    return await response.json()
        else:
            async with self.aiosession.get(search_url, headers=headers) as response:
                return await response.json()

    def get_relevant_documents(self, query: str) -> List[Document]:
        search_results = self._search(query)

        return [
            Document(page_content=result["content"], metadata=result)
            for result in search_results
        ]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        search_results = await self._async_search(query)

        return [
            Document(page_content=result["content"], metadata=result)
            for result in search_results["value"]
        ]
