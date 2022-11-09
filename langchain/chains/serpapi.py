"""Chain that calls SerpAPI.

Heavily borrowed from https://github.com/ofirpress/self-ask
"""
import os
import sys
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, root_validator

from langchain.chains.base import Chain


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


class SerpAPIChain(Chain, BaseModel):
    """Chain that calls SerpAPI.

    To use, you should have the ``google-search-results`` python package installed,
    and the environment variable ``SERPAPI_API_KEY`` set with your API key, or pass
    `serpapi_api_key` as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain import SerpAPIChain
            serpapi = SerpAPIChain()
    """

    search_engine: Any  #: :meta private:
    input_key: str = "search_query"  #: :meta private:
    output_key: str = "search_result"  #: :meta private:

    serpapi_api_key: Optional[str] = os.environ.get("SERPAPI_API_KEY")

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @property
    def input_keys(self) -> List[str]:
        """Return the singular input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.

        :meta private:
        """
        return [self.output_key]

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        serpapi_api_key = values.get("serpapi_api_key")

        if serpapi_api_key is None or serpapi_api_key == "":
            raise ValueError(
                "Did not find SerpAPI API key, please add an environment variable"
                " `SERPAPI_API_KEY` which contains it, or pass `serpapi_api_key` "
                "as a named parameter to the constructor."
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
            "api_key": self.serpapi_api_key,
            "engine": "google",
            "q": inputs[self.input_key],
            "google_domain": "google.com",
            "gl": "us",
            "hl": "en",
        }
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
        elif "snippet" in res["organic_results"][0].keys():
            toret = res["organic_results"][0]["snippet"]
        else:
            toret = None
        return {self.output_key: toret}

    def search(self, search_question: str) -> str:
        """Run search query against SerpAPI.

        Args:
            search_question: Question to run against the SerpAPI.

        Returns:
            Answer from the search engine.

        Example:
            .. code-block:: python

                answer = serpapi.search("What is the capital of Idaho?")
        """
        return self({self.input_key: search_question})[self.output_key]
