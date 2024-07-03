"""Util that calls Google Search using the Serper.dev API."""

from typing import Any, Dict, List, Optional

import aiohttp
import requests
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.utils import get_from_dict_or_env
from typing_extensions import Literal


class GoogleSerperAPIWrapper(BaseModel):
    """Wrapper around the Serper.dev Google Search API.

    You can create a free API key at https://serper.dev.

    To use, you should have the environment variable ``SERPER_API_KEY``
    set with your API key, or pass `serper_api_key` as a named parameter
    to the constructor.

    Example:
        .. code-block:: python

            from langchain_community.utilities import GoogleSerperAPIWrapper
            google_serper = GoogleSerperAPIWrapper()
    """

    k: int = 10
    gl: str = "us"
    hl: str = "en"
    # "places" and "images" is available from Serper but not implemented in the
    # parser of run(). They can be used in results()
    type: Literal["news", "search", "places", "images"] = "search"
    result_key_for_type = {
        "news": "news",
        "places": "places",
        "images": "images",
        "search": "organic",
    }

    tbs: Optional[str] = None
    serper_api_key: Optional[str] = None
    aiosession: Optional[aiohttp.ClientSession] = None

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        serper_api_key = get_from_dict_or_env(
            values, "serper_api_key", "SERPER_API_KEY"
        )
        values["serper_api_key"] = serper_api_key

        return values

    def results(self, query: str, **kwargs: Any) -> Dict:
        """Run query through GoogleSearch."""
        return self._google_serper_api_results(
            query,
            gl=self.gl,
            hl=self.hl,
            num=self.k,
            tbs=self.tbs,
            search_type=self.type,
            **kwargs,
        )

    def run(self, query: str, **kwargs: Any) -> str:
        """Run query through GoogleSearch and parse result."""
        results = self._google_serper_api_results(
            query,
            gl=self.gl,
            hl=self.hl,
            num=self.k,
            tbs=self.tbs,
            search_type=self.type,
            **kwargs,
        )

        return self._parse_results(results)

    async def aresults(self, query: str, **kwargs: Any) -> Dict:
        """Run query through GoogleSearch."""
        results = await self._async_google_serper_search_results(
            query,
            gl=self.gl,
            hl=self.hl,
            num=self.k,
            search_type=self.type,
            tbs=self.tbs,
            **kwargs,
        )
        return results

    async def arun(self, query: str, **kwargs: Any) -> str:
        """Run query through GoogleSearch and parse result async."""
        results = await self._async_google_serper_search_results(
            query,
            gl=self.gl,
            hl=self.hl,
            num=self.k,
            search_type=self.type,
            tbs=self.tbs,
            **kwargs,
        )

        return self._parse_results(results)

    def _parse_snippets(self, results: dict) -> List[str]:
        snippets = []

        if results.get("answerBox"):
            answer_box = results.get("answerBox", {})
            if answer_box.get("answer"):
                return [answer_box.get("answer")]
            elif answer_box.get("snippet"):
                return [answer_box.get("snippet").replace("\n", " ")]
            elif answer_box.get("snippetHighlighted"):
                return answer_box.get("snippetHighlighted")

        if results.get("knowledgeGraph"):
            kg = results.get("knowledgeGraph", {})
            title = kg.get("title")
            entity_type = kg.get("type")
            if entity_type:
                snippets.append(f"{title}: {entity_type}.")
            description = kg.get("description")
            if description:
                snippets.append(description)
            for attribute, value in kg.get("attributes", {}).items():
                snippets.append(f"{title} {attribute}: {value}.")

        for result in results[self.result_key_for_type[self.type]][: self.k]:
            if "snippet" in result:
                snippets.append(result["snippet"])
            for attribute, value in result.get("attributes", {}).items():
                snippets.append(f"{attribute}: {value}.")

        if len(snippets) == 0:
            return ["No good Google Search Result was found"]
        return snippets

    def _parse_results(self, results: dict) -> str:
        return " ".join(self._parse_snippets(results))

    def _google_serper_api_results(
        self, search_term: str, search_type: str = "search", **kwargs: Any
    ) -> dict:
        headers = {
            "X-API-KEY": self.serper_api_key or "",
            "Content-Type": "application/json",
        }
        params = {
            "q": search_term,
            **{key: value for key, value in kwargs.items() if value is not None},
        }
        response = requests.post(
            f"https://google.serper.dev/{search_type}", headers=headers, params=params
        )
        response.raise_for_status()
        search_results = response.json()
        return search_results

    async def _async_google_serper_search_results(
        self, search_term: str, search_type: str = "search", **kwargs: Any
    ) -> dict:
        headers = {
            "X-API-KEY": self.serper_api_key or "",
            "Content-Type": "application/json",
        }
        url = f"https://google.serper.dev/{search_type}"
        params = {
            "q": search_term,
            **{key: value for key, value in kwargs.items() if value is not None},
        }

        if not self.aiosession:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, params=params, headers=headers, raise_for_status=False
                ) as response:
                    search_results = await response.json()
        else:
            async with self.aiosession.post(
                url, params=params, headers=headers, raise_for_status=True
            ) as response:
                search_results = await response.json()

        return search_results
