"""Util that calls Naver Search API.

In order to set this up, follow instructions at:
https://developers.naver.com/docs/serviceapi/search/news/news.md
"""

import json
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.parse

import aiohttp
import requests
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, SecretStr, model_validator

NAVER_API_URL = "https://openapi.naver.com/v1/search"


class NaverSearchAPIWrapper(BaseModel):
    """Wrapper for Naver Search API."""

    naver_client_id: SecretStr
    naver_client_secret: SecretStr

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and endpoint exists in environment."""
        naver_client_id = get_from_dict_or_env(
            values, "naver_client_id", "NAVER_CLIENT_ID"
        )
        naver_client_secret = get_from_dict_or_env(
            values, "naver_client_secret", "NAVER_CLIENT_SECRET"
        )
        values["naver_client_id"] = naver_client_id
        values["naver_client_secret"] = naver_client_secret

        return values

    def raw_results(
        self,
        query: str,
        search_type: str = "news",
        display: Optional[int] = 10,
        start: Optional[int] = 1,
        sort: Optional[str] = "sim",  # sim (similarity) or date
    ) -> Dict:
        """Get raw results from the Naver Search API."""
        enc_text = urllib.parse.quote(query)
        url = f"{NAVER_API_URL}/{search_type}?query={enc_text}&display={display}&start={start}&sort={sort}"
        
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", self.naver_client_id.get_secret_value())
        request.add_header("X-Naver-Client-Secret", self.naver_client_secret.get_secret_value())
        
        response = urllib.request.urlopen(request)
        response_code = response.getcode()
        
        if response_code == 200:
            response_body = response.read().decode('utf-8')
            return json.loads(response_body)
        else:
            raise Exception(f"Error Code: {response_code}")

    def results(
        self,
        query: str,
        search_type: str = "news",
        display: Optional[int] = 10,
        start: Optional[int] = 1,
        sort: Optional[str] = "sim",
    ) -> List[Dict]:
        """Run query through Naver Search and return cleaned results.

        Args:
            query: The query to search for.
            search_type: The type of search (news, blog, webkr, etc.)
            display: The number of results to return (max 100).
            start: The starting position for results.
            sort: The sort order (sim for similarity, date for date).
            
        Returns:
            A list of dictionaries containing the cleaned search results.
        """
        raw_search_results = self.raw_results(
            query,
            search_type=search_type,
            display=display,
            start=start,
            sort=sort,
        )
        return self.clean_results(raw_search_results["items"])

    async def raw_results_async(
        self,
        query: str,
        search_type: str = "news",
        display: Optional[int] = 10,
        start: Optional[int] = 1,
        sort: Optional[str] = "sim",
    ) -> Dict:
        """Get results from the Naver Search API asynchronously."""
        enc_text = urllib.parse.quote(query)
        url = f"{NAVER_API_URL}/{search_type}?query={enc_text}&display={display}&start={start}&sort={sort}"

        async def fetch() -> str:
            headers = {
                "X-Naver-Client-Id": self.naver_client_id.get_secret_value(),
                "X-Naver-Client-Secret": self.naver_client_secret.get_secret_value()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.text()
                        return data
                    else:
                        raise Exception(f"Error {response.status}: {response.reason}")

        results_json_str = await fetch()
        return json.loads(results_json_str)

    async def results_async(
        self,
        query: str,
        search_type: str = "news",
        display: Optional[int] = 10,
        start: Optional[int] = 1,
        sort: Optional[str] = "sim",
    ) -> List[Dict]:
        """Get cleaned results from Naver Search API asynchronously."""
        results_json = await self.raw_results_async(
            query=query,
            search_type=search_type,
            display=display,
            start=start,
            sort=sort,
        )
        return self.clean_results(results_json["items"])

    def clean_results(self, results: List[Dict]) -> List[Dict]:
        """Clean results from Naver Search API."""
        clean_results = []
        for result in results:
            # Remove HTML tags from title and description
            title = result.get("title", "").replace("<b>", "").replace("</b>", "")
            description = result.get("description", "").replace("<b>", "").replace("</b>", "")
            
            clean_result = {
                "title": title,
                "link": result.get("link", ""),
                "description": description,
            }
            
            # Add optional fields if they exist
            for field in ["bloggername", "postdate", "pubDate"]:
                if field in result:
                    clean_result[field] = result[field]
                    
            clean_results.append(clean_result)
        return clean_results
