"""Util that calls Google Scholar Search."""
from typing import Any, Dict, Optional, cast

from langchain_core.pydantic_v1 import BaseModel, Extra, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env


class GoogleJobsAPIWrapper(BaseModel):
    """Wrapper for SerpApi's Google Scholar API

    You can create SerpApi.com key by signing up at: https://serpapi.com/users/sign_up.
    The wrapper uses the SerpApi.com python package:
    https://serpapi.com/integrations/python
    To use, you should have the environment variable ``SERPAPI_API_KEY``
    set with your API key, or pass `serp_api_key` as a named parameter
    to the constructor.
     Example:
        .. code-block:: python
        from langchain_community.utilities import GoogleJobsAPIWrapper
        google_Jobs = GoogleJobsAPIWrapper()
        google_Jobs.run('langchain')
    """

    serp_search_engine: Any
    serp_api_key: Optional[SecretStr] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["serp_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "serp_api_key", "SERPAPI_API_KEY")
        )

        try:
            from serpapi import SerpApiClient

        except ImportError:
            raise ImportError(
                "google-search-results is not installed. "
                "Please install it with `pip install google-search-results"
                ">=2.4.2`"
            )
        serp_search_engine = SerpApiClient
        values["serp_search_engine"] = serp_search_engine

        return values

    def run(self, query: str) -> str:
        """Run query through Google Trends with Serpapi"""

        # set up query
        serpapi_api_key = cast(SecretStr, self.serp_api_key)
        params = {
            "engine": "google_jobs",
            "api_key": serpapi_api_key.get_secret_value(),
            "q": query,
        }

        total_results = []
        client = self.serp_search_engine(params)
        total_results = client.get_dict()["jobs_results"]

        # extract 1 job info:
        res_str = ""
        for i in range(1):
            job = total_results[i]
            res_str += (
                "\n_______________________________________________"
                + f"\nJob Title: {job['title']}\n"
                + f"Company Name: {job['company_name']}\n"
                + f"Location: {job['location']}\n"
                + f"Description: {job['description']}"
                + "\n_______________________________________________\n"
            )

        return res_str + "\n"
