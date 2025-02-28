"""Util that calls Tavily Search API.

In order to set this up, follow instructions at:
https://docs.tavily.com/docs/tavily-api/introduction
"""

import json
from typing import Any, Dict, List, Literal, Optional

import aiohttp
import requests
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, SecretStr, model_validator

TAVILY_API_URL = "https://api.tavily.com"


class TavilySearchAPIWrapper(BaseModel):
    """Wrapper for Tavily Search API."""

    tavily_api_key: SecretStr

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and endpoint exists in environment."""
        tavily_api_key = get_from_dict_or_env(
            values, "tavily_api_key", "TAVILY_API_KEY"
        )
        values["tavily_api_key"] = tavily_api_key

        return values

    def raw_results(
        self,
        query: str,
        max_results: Optional[int] = None,
        search_depth: Optional[Literal["basic", "advanced"]] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        include_answer: Optional[bool] = None,
        include_raw_content: Optional[bool] = None,
        include_images: Optional[bool] = None,
        include_image_descriptions: Optional[bool] = None,
        topic: Optional[Literal["general", "news"]] = None,
        time_range: Optional[
            Literal["day", "week", "month", "year", "d", "w", "m", "y"]
        ] = None,
        days: Optional[int] = None,
    ) -> Dict:
        params = {
            "api_key": self.tavily_api_key.get_secret_value(),
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_domains": include_domains,
            "exclude_domains": exclude_domains,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
            "include_images": include_images,
            "include_image_descriptions": include_image_descriptions,
            "topic": topic,
            **({"time_range": time_range} if time_range is not None else {}),
            **({"days": days} if days is not None else {}),
        }
        from pdb import set_trace as bp
        bp()
        response = requests.post(
            # type: ignore
            f"{TAVILY_API_URL}/search",
            json=params,
        )
        if response.status_code != 200:
            detail = response.json().get("detail", {})
            error_message = (
                detail.get("error") if isinstance(detail, dict) else "Unknown error"
            )
            raise ValueError(f"Error {response.status_code}: {error_message}")
        return response.json()

    async def raw_results_async(
        self,
        query: str,
        max_results: Optional[int] = 5,
        search_depth: Optional[Literal["basic", "advanced"]] = "basic",
        include_domains: Optional[List[str]] = [],
        exclude_domains: Optional[List[str]] = [],
        include_answer: Optional[bool] = False,
        include_raw_content: Optional[bool] = False,
        include_images: Optional[bool] = False,
        include_image_descriptions: Optional[bool] = False,
        topic: Optional[Literal["general", "news"]] = "general",
        time_range: Optional[
            Literal["day", "week", "month", "year", "d", "w", "m", "y"]
        ] = None,
        days: Optional[int] = None,
    ) -> Dict:
        """Get results from the Tavily Search API asynchronously."""

        # Function to perform the API call
        async def fetch() -> str:
            params = {
                "api_key": self.tavily_api_key.get_secret_value(),
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
                "include_domains": include_domains,
                "exclude_domains": exclude_domains,
                "include_answer": include_answer,
                "include_raw_content": include_raw_content,
                "include_images": include_images,
                "include_image_descriptions": include_image_descriptions,
                "topic": topic,
                **({"time_range": time_range} if time_range is not None else {}),
                **({"days": days} if days is not None else {}),
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{TAVILY_API_URL}/search", json=params) as res:
                    if res.status == 200:
                        data = await res.text()
                        return data
                    else:
                        raise Exception(f"Error {res.status}: {res.reason}")

        results_json_str = await fetch()

        return json.loads(results_json_str)
