"""Util that calls Google Search."""
from typing import Dict, Optional

from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from langchain.utils import get_from_dict_or_env


class GoogleScholarAPIWrapper(BaseModel):
    """Wrapper for Google Scholar API

    You can create serpapi key by signing up at: https://serpapi.com/users/sign_up.

    The wrapper uses the serpapi python package:
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

    top_k_results: int = 10
    hl: str = "en"
    lr: str = "lang_en"
    serp_api_key: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        serp_api_key = get_from_dict_or_env(values, "serp_api_key", "SERP_API_KEY")
        values["SERP_API_KEY"] = serp_api_key

        try:
            from serpapi import GoogleScholarSearch

        except ImportError:
            raise ImportError(
                "google-search-results is not installed. "
                "Please install it with `pip install google-search-results"
                ">=2.4.2`"
            )
        GoogleScholarSearch.SERP_API_KEY = serp_api_key
        values["google_scholar_engine"] = GoogleScholarSearch

        return values

    def run(self, query: str) -> str:
        """Run query through GoogleSearchScholar and parse result"""
        total_results = []
        p = 0
        for p in range(1, self.top_k_results // 10 + 1):
            # by default every page returns 10 results, therefore to get top_k results
            # we need to get results from first k/10 pages
            results = self.google_scholar_engine(  # type: ignore
                {"q": query, "start": p, "hl": self.hl, "lr": self.lr}
            ).get_dict()
            total_results.extend(results.get("organic_results"))
        if (
            self.top_k_results % 10 != 0
        ):  # from the last page we would only need k%10 results
            # if k is not divisble by 10,
            results = self.google_scholar_engine(  # type: ignore
                {
                    "q": query,
                    "start": p + 1,
                    "num": self.top_k_results % 10,
                    "hl": self.hl,
                    "lr": self.lr,
                }
            ).get_dict()
            total_results.extend(results.get("organic_results"))
        if not results:
            return "No good Google Scholar Result was found"
        docs = [
            f"Title: {result.get('title','')}\n"
            f"Authors: {','.join([author.get('name') for author in result.get('publication_info',{}).get('authors',[])])}\n"  # noqa: E501
            f"Summary: {result.get('publication_info',{}).get('summary','')}\n"
            f"Total-Citations: {result.get('inline_links',{}).get('cited_by',{}).get('total','')}"  # noqa: E501
            for result in total_results
        ]
        return "\n\n".join(docs)
