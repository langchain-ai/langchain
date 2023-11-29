from typing import Any, Dict, Optional

import aiohttp
import requests
from langchain_core.pydantic_v1 import BaseModel, root_validator

from langchain.utils import get_from_dict_or_env


class SearchApiAPIWrapper(BaseModel):
    """
    Wrapper around SearchApi API.

    To use, you should have the environment variable ``SEARCHAPI_API_KEY``
    set with your API key, or pass `searchapi_api_key`
    as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain.utilities import SearchApiAPIWrapper
            searchapi = SearchApiAPIWrapper()
    """

    # Use "google" engine by default.
    # Full list of supported ones can be found in https://www.searchapi.io docs
    engine: str = "google"
    searchapi_api_key: Optional[str] = None
    aiosession: Optional[aiohttp.ClientSession] = None

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that API key exists in environment."""
        searchapi_api_key = get_from_dict_or_env(
            values, "searchapi_api_key", "SEARCHAPI_API_KEY"
        )
        values["searchapi_api_key"] = searchapi_api_key
        return values

    def run(self, query: str, **kwargs: Any) -> str:
        results = self.results(query, **kwargs)
        return self._result_as_string(results)

    async def arun(self, query: str, **kwargs: Any) -> str:
        results = await self.aresults(query, **kwargs)
        return self._result_as_string(results)

    def results(self, query: str, **kwargs: Any) -> dict:
        results = self._search_api_results(query, **kwargs)
        return results

    async def aresults(self, query: str, **kwargs: Any) -> dict:
        results = await self._async_search_api_results(query, **kwargs)
        return results

    def _prepare_request(self, query: str, **kwargs: Any) -> dict:
        return {
            "url": "https://www.searchapi.io/api/v1/search",
            "headers": {
                "Authorization": f"Bearer {self.searchapi_api_key}",
            },
            "params": {
                "engine": self.engine,
                "q": query,
                **{key: value for key, value in kwargs.items() if value is not None},
            },
        }

    def _search_api_results(self, query: str, **kwargs: Any) -> dict:
        request_details = self._prepare_request(query, **kwargs)
        response = requests.get(
            url=request_details["url"],
            params=request_details["params"],
            headers=request_details["headers"],
        )
        response.raise_for_status()
        return response.json()

    async def _async_search_api_results(self, query: str, **kwargs: Any) -> dict:
        """Use aiohttp to send request to SearchApi API and return results async."""
        request_details = self._prepare_request(query, **kwargs)
        if not self.aiosession:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url=request_details["url"],
                    headers=request_details["headers"],
                    params=request_details["params"],
                    raise_for_status=True,
                ) as response:
                    results = await response.json()
        else:
            async with self.aiosession.get(
                url=request_details["url"],
                headers=request_details["headers"],
                params=request_details["params"],
                raise_for_status=True,
            ) as response:
                results = await response.json()
        return results

    @staticmethod
    def _result_as_string(result: dict) -> str:
        toret = "No good search result found"
        if "answer_box" in result.keys() and "answer" in result["answer_box"].keys():
            toret = result["answer_box"]["answer"]
        elif "answer_box" in result.keys() and "snippet" in result["answer_box"].keys():
            toret = result["answer_box"]["snippet"]
        elif "knowledge_graph" in result.keys():
            toret = result["knowledge_graph"]["description"]
        elif "organic_results" in result.keys():
            snippets = [
                r["snippet"] for r in result["organic_results"] if "snippet" in r.keys()
            ]
            toret = "\n".join(snippets)
        elif "jobs" in result.keys():
            jobs = [
                r["description"] for r in result["jobs"] if "description" in r.keys()
            ]
            toret = "\n".join(jobs)
        elif "videos" in result.keys():
            videos = [
                f"""Title: "{r["title"]}" Link: {r["link"]}"""
                for r in result["videos"]
                if "title" in r.keys()
            ]
            toret = "\n".join(videos)
        elif "images" in result.keys():
            images = [
                f"""Title: "{r["title"]}" Link: {r["original"]["link"]}"""
                for r in result["images"]
                if "original" in r.keys()
            ]
            toret = "\n".join(images)
        return toret
