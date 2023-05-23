"""Retriever wrapper for Azure Cognitive Search."""
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

    service_name: str = ""
    """Name of Azure Cognitive Search service"""
    index_name: str = ""
    """Name of Index inside Azure Cognitive Search service"""
    api_key: str = ""
    """Required. Version of the REST API used for the request. For a list of supported versions, see API versions. For this operation, the api-version is specified as a URI parameter regardless of whether you call Search Documents with GET or POST."""
    api_version: str = "2021-04-30-Preview"
    """API version"""
    aiosession: Optional[aiohttp.ClientSession] = None
    """ClientSession, in case we want to reuse connection for better performance."""
    content_key: str = "content"
    """Key in a retrieved result to set as the Document page_content."""

    query_parameters: Optional[Dict] = None
    """Optional. Parameters to be passed to the search query.
    
    For more information see: https://learn.microsoft.com/en-us/rest/api/searchservice/search-documents
    
    {
        "count": true | false (default),  
        "facets": [ "facet_expression_1", "facet_expression_2", ... ],  
        "filter": "odata_filter_expression",  
        "highlight": "highlight_field_1, highlight_field_2, ...",  
        "highlightPreTag": "pre_tag",  
        "highlightPostTag": "post_tag",  
        "minimumCoverage": # (% of index that must be covered to declare query successful; default 100),  
        "orderby": "orderby_expression",  
        "queryType": "simple" (default) | "full",
        "scoringParameters": [ "scoring_parameter_1", "scoring_parameter_2", ... ],  
        "scoringProfile": "scoring_profile_name",  
        "scoringStatistics" : "local" | "global",
        "search": "simple_query_expression",  
        "searchFields": "field_name_1, field_name_2, ...",  
        "searchMode": "any" (default) | "all",  
        "select": "field_name_1, field_name_2, ...",  
        "sessionId" : "session_id",
        "skip": # (default 0),  
        "top": #  
    }
    """

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

    def _build_search_url(self, query: str) -> str:
        base_url = f"https://{self.service_name}.search.windows.net/"
        # iterate over query parameters and add them to the url
        endpoint_path = f"indexes/{self.index_name}/docs?api-version={self.api_version}"
        query = base_url + endpoint_path + f"&search={query}"
        if self.query_parameters:
            for key, value in self.query_parameters.items():
                query += f"&${key}={value}"
        return query

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }

    def _search(self, query: str) -> List[dict]:
        search_url = self._build_search_url(query)
        response = requests.get(search_url, headers=self._headers)
        if response.status_code != 200:
            raise Exception(f"Error in search request: {response}")

        return json.loads(response.text)["value"]

    async def _asearch(self, query: str) -> List[dict]:
        search_url = self._build_search_url(query)
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

    def get_relevant_documents(self, query: str) -> List[Document]:
        search_results = self._search(query)

        return [
            Document(page_content=result.pop(self.content_key), metadata=result)
            for result in search_results
        ]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        search_results = await self._asearch(query)

        return [
            Document(page_content=result.pop(self.content_key), metadata=result)
            for result in search_results
        ]
