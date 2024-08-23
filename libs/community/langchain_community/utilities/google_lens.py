"""Util that calls Google Lens Search."""

from typing import Any, Dict, Optional, cast

import requests
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env


class GoogleLensAPIWrapper(BaseModel):
    """Wrapper for SerpApi's Google Lens API

    You can create SerpApi.com key by signing up at: https://serpapi.com/users/sign_up.

    The wrapper uses the SerpApi.com python package:
    https://serpapi.com/integrations/python

    To use, you should have the environment variable ``SERPAPI_API_KEY``
    set with your API key, or pass `serp_api_key` as a named parameter
    to the constructor.

     Example:
        .. code-block:: python

        from langchain_community.utilities import GoogleLensAPIWrapper
        google_lens = GoogleLensAPIWrapper()
        google_lens.run('langchain')
    """

    serp_search_engine: Any
    serp_api_key: Optional[SecretStr] = None

    class Config:
        extra = "forbid"

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["serp_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "serp_api_key", "SERPAPI_API_KEY")
        )

        return values

    def run(self, query: str) -> str:
        """Run query through Google Trends with Serpapi"""
        serpapi_api_key = cast(SecretStr, self.serp_api_key)

        params = {
            "engine": "google_lens",
            "api_key": serpapi_api_key.get_secret_value(),
            "url": query,
        }
        queryURL = f"https://serpapi.com/search?engine={params['engine']}&api_key={params['api_key']}&url={params['url']}"
        response = requests.get(queryURL)

        if response.status_code != 200:
            return "Google Lens search failed"

        responseValue = response.json()

        if responseValue["search_metadata"]["status"] != "Success":
            return "Google Lens search failed"

        xs = ""
        if (
            "knowledge_graph" in responseValue
            and len(responseValue["knowledge_graph"]) > 0
        ):
            subject = responseValue["knowledge_graph"][0]
            xs += f"Subject:{subject['title']}({subject['subtitle']})\n"
            xs += f"Link to subject:{subject['link']}\n\n"
        xs += "Related Images:\n\n"
        for image in responseValue["visual_matches"]:
            xs += f"Title: {image['title']}\n"
            xs += f"Source({image['source']}): {image['link']}\n"
            xs += f"Image: {image['thumbnail']}\n\n"
        if "reverse_image_search" in responseValue:
            xs += (
                "Reverse Image Search"
                + f"Link: {responseValue['reverse_image_search']['link']}\n"
            )
        print(xs)  # noqa: T201

        docs = [xs]

        return "\n\n".join(docs)
