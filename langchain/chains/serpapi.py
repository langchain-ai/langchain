"""Chain that calls SerpAPI.

Heavily borrowed from https://github.com/ofirpress/self-ask
"""
import os
from typing import Any, Dict, List

from pydantic import BaseModel, Extra, root_validator

from langchain.chains.base import Chain


class SerpAPIChain(Chain, BaseModel):
    """Chain that calls SerpAPI."""

    search_engine: Any
    input_key: str = "search_query"
    output_key: str = "search_result"

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @property
    def input_keys(self) -> List[str]:
        """Return the singular input key."""
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key."""
        return [self.output_key]

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        if "SERPAPI_API_KEY" not in os.environ:
            raise ValueError(
                "Did not find SerpAPI API key, please add an environment variable"
                " `SERPAPI_API_KEY` which contains it."
            )
        try:
            from serpapi import GoogleSearch

            values["search_engine"] = GoogleSearch
        except ImportError:
            raise ValueError(
                "Could not import serpapi python package. "
                "Please it install it with `pip install google-search-results`."
            )
        return values

    def _run(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        params = {
            "api_key": os.environ["SERPAPI_API_KEY"],
            "engine": "google",
            "q": inputs[self.input_key],
            "google_domain": "google.com",
            "gl": "us",
            "hl": "en",
        }

        search = self.search_engine(params)
        res = search.get_dict()

        if "answer_box" in res.keys() and "answer" in res["answer_box"].keys():
            toret = res["answer_box"]["answer"]
        elif "answer_box" in res.keys() and "snippet" in res["answer_box"].keys():
            toret = res["answer_box"]["snippet"]
        elif (
            "answer_box" in res.keys()
            and "snippet_highlighted_words" in res["answer_box"].keys()
        ):
            toret = res["answer_box"]["snippet_highlighted_words"][0]
        elif "snippet" in res["organic_results"][0].keys():
            toret = res["organic_results"][0]["snippet"]
        else:
            toret = None
        return {self.output_key: toret}

    def search(self, search_question: str) -> str:
        """More user-friendly interface for interfacing with search."""
        return self({self.input_key: search_question})[self.output_key]
