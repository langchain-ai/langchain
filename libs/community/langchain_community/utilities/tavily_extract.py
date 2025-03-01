"""Util that calls Tavily Extract API.

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


class TavilyExtractAPIWrapper(BaseModel):
    """Wrapper for Tavily Extract API."""

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
        urls: List[str],
        extract_depth: Optional[
            Literal["basic", "advanced"]
        ] = "advanced",  
        include_images: Optional[bool] = False,
    ) -> Dict:
        params = {
            "api_key": self.tavily_api_key.get_secret_value(),
            "urls": urls,
            "include_images": include_images,
            "extract_depth": extract_depth,
        }

        response = requests.post(
            # type: ignore
            f"{TAVILY_API_URL}/extract",
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
        urls: List[str],
        include_images: Optional[bool] = False,
        extract_depth: Optional[
            Literal["basic", "advanced"]
        ] = "advanced",
    ) -> Dict:
        """Get results from the Tavily Extract API asynchronously."""

        # Function to perform the API call
        async def fetch() -> str:
            params = {
                "api_key": self.tavily_api_key.get_secret_value(),
                "urls": urls,
                "include_images": include_images,
                "extract_depth": extract_depth,
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{TAVILY_API_URL}/extract", json=params
                ) as res:
                    if res.status == 200:
                        data = await res.text()
                        return data
                    else:
                        raise Exception(f"Error {res.status}: {res.reason}")

        results_json_str = await fetch()

        return json.loads(results_json_str)
