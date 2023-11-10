"""Util that calls Google Scholar Search."""
from typing import Any, Dict, Optional, cast

from langchain.pydantic_v1 import BaseModel, Extra, SecretStr, root_validator
from langchain.utils import convert_to_secret_str, get_from_dict_or_env


class GoogleScholarAPIWrapper(BaseModel):
    """Wrapper for SerpApi's Google Scholar API

    You can create SerpApi.com key by signing up at: https://serpapi.com/users/sign_up.

    The wrapper uses the SerpApi.com python package:
    https://serpapi.com/integrations/python#search-google-scholar

    To use, you should have the environment variable ``SERP_API_KEY``
    set with your API key, or pass `serp_api_key` as a named parameter
    to the constructor.

    Attributes:
        top_k_results: number of results to return from google-scholar query search.
            By default it returns top 10 results.
        hl: attribute defines the language to use for the Google Scholar search.
            It's a two-letter language code.
            (e.g., en for English, es for Spanish, or fr for French). Head to the
            Google languages page for a full list of supported Google languages:
            https://serpapi.com/google-languages

        lr: attribute defines one or multiple languages to limit the search to.
            It uses lang_{two-letter language code} to specify languages
            and | as a delimiter. (e.g., lang_fr|lang_de will only search French
            and German pages). Head to the Google lr languages for a full
            list of supported languages: https://serpapi.com/google-lr-languages

     Example:
        .. code-block:: python

        from langchain.utilities import GoogleScholarAPIWrapper
        google_scholar = GoogleScholarAPIWrapper()
        google_scholar.run('langchain')
    """

    search_engine: Any  #: :meta private:

    top_k_results: int = 10
    hl: str = "en"
    lr: str = "lang_en"
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

        values["search_engine"] = SerpApiClient

        return values

    def run(self, query: str) -> str:
        """Run query through SerpApi.com and parse result"""
        total_results = []

        serpapi_api_key = cast(SecretStr, self.serp_api_key)
        params = {
            "engine": "google_scholar",
            "api_key": serpapi_api_key.get_secret_value(),
            "q": query,
            "hl": self.hl,
            "num": min(self.top_k_results, 20),  # if top_k_result is less than 20.
            "lr": self.lr,
        }

        client = self.search_engine(params)

        for page in client.pagination():
            results = page.get("organic_results", [])
            total_results.extend(results)

        if not total_results:
            return "No good Google Scholar Result was found"

        docs = [
            f"Title: {result.get('title','')}\n"
            f"Authors: {','.join([author.get('name') for author in result.get('publication_info',{}).get('authors',[])])}\n"  # noqa: E501
            f"Summary: {result.get('publication_info',{}).get('summary','')}\n"
            f"Total-Citations: {result.get('inline_links',{}).get('cited_by',{}).get('total','')}"  # noqa: E501
            for result in total_results
        ]
        return "\n\n".join(docs)
