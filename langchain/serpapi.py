"""Chain that calls SerpAPI.

Heavily borrowed from https://github.com/ofirpress/self-ask
"""
import os
import sys
from typing import Any, Dict, Optional

from pydantic import BaseModel, Extra, Field, root_validator

from langchain.utils import get_from_dict_or_env


class HiddenPrints:
    """Context manager to hide prints."""

    def __enter__(self) -> None:
        """Open file to pipe stdout to."""
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, *_: Any) -> None:
        """Close file that stdout was piped to."""
        sys.stdout.close()
        sys.stdout = self._original_stdout


def _get_default_params() -> dict:
    return {
        "engine": "google",
        "google_domain": "google.com",
        "gl": "us",
        "hl": "en",
    }


class SerpAPIWrapper(BaseModel):
    """Wrapper around SerpAPI.

    To use, you should have the ``google-search-results`` python package installed,
    and the environment variable ``SERPAPI_API_KEY`` set with your API key, or pass
    `serpapi_api_key` as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain import SerpAPIWrapper
            serpapi = SerpAPIWrapper()
    """

    search_engine: Any  #: :meta private:
    params: dict = Field(default_factory=_get_default_params)
    serpapi_api_key: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        serpapi_api_key = get_from_dict_or_env(
            values, "serpapi_api_key", "SERPAPI_API_KEY"
        )
        values["serpapi_api_key"] = serpapi_api_key
        try:
            from serpapi import GoogleSearch

            values["search_engine"] = GoogleSearch
        except ImportError:
            raise ValueError(
                "Could not import serpapi python package. "
                "Please it install it with `pip install google-search-results`."
            )
        return values

    def run(self, query: str) -> str:
        """Run query through SerpAPI and parse result."""
        _params = {
            "api_key": self.serpapi_api_key,
            "q": query,
        }
        params = {**self.params, **_params}
        with HiddenPrints():
            search = self.search_engine(params)
            res = search.get_dict()
        if "error" in res.keys():
            raise ValueError(f"Got error from SerpAPI: {res['error']}")
        if "answer_box" in res.keys() and "answer" in res["answer_box"].keys():
            toret = res["answer_box"]["answer"]
        elif "answer_box" in res.keys() and "snippet" in res["answer_box"].keys():
            toret = res["answer_box"]["snippet"]
        elif (
            "answer_box" in res.keys()
            and "snippet_highlighted_words" in res["answer_box"].keys()
        ):
            toret = res["answer_box"]["snippet_highlighted_words"][0]
        elif (
            "sports_results" in res.keys()
            and "game_spotlight" in res["sports_results"].keys()
        ):
            toret = res["sports_results"]["game_spotlight"]
        elif (
            "knowledge_graph" in res.keys()
            and "description" in res["knowledge_graph"].keys()
        ):
            toret = res["knowledge_graph"]["description"]
        elif "snippet" in res["organic_results"][0].keys():
            toret = res["organic_results"][0]["snippet"]

        else:
            toret = "No good search result found"
        return toret


# For backwards compatability

SerpAPIChain = SerpAPIWrapper
