"""Util that calls you.com Search API.

In order to set this up, follow instructions at:
"""
import json
from typing import Any, Dict, List, Literal, Optional

import aiohttp
import requests
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.utils import get_from_dict_or_env

YOU_API_URL = "https://api.ydc-index.io"


class YouHitMetadata(BaseModel):
    """Metadata on a single hit from you.com"""

    title: str = Field(description="The title of the result")
    url: str = Field(description="The url of the result")
    thumbnail_url: str = Field(description="Thumbnail associated with the result")
    description: str = Field(description="Details about the result")


class YouHit(YouHitMetadata):
    """A single hit from you.com, which may contain multiple snippets"""

    snippets: List[str] = Field(description="One or snippets of text")


class YouAPIOutput(BaseModel):
    """The output from you.com api"""

    hits: List[YouHit] = Field(
        description="A list of dictionaries containing the results"
    )


class YouDocument(BaseModel):
    """The output of parsing one snippet"""

    page_content: str = Field(description="One snippet of text")
    metadata: YouHitMetadata


class YouSearchAPIWrapper(BaseModel):
    """Wrapper for you.com Search API.

    To connect to the You.com api requires an API key which
    you can get at https://api.you.com.
    You can check out the docs at https://documentation.you.com.

    You need to set the environment variable `YDC_API_KEY` for retriever to operate.

    Attributes
    ----------
    ydc_api_key: str, optional
        you.com api key, if YDC_API_KEY is not set in the environment
    num_web_results: int, optional
        The max number of web results to return, must be under 20
    safesearch: str, optional
        Safesearch settings, one of off, moderate, strict, defaults to moderate
    country: str, optional
        Country code, ex: 'US' for united states, see api docs for list
    k: int, optional
        max number of Documents to return using `results()`
    n_hits: int, optional, deprecated
        Alias for num_web_results
    n_snippets_per_hit: int, optional
        limit the number of snippets returned per hit
    endpoint_type: str, optional
        you.com endpoints: search, news, rag;
        `web` and `snippet` alias `search`
        `rag` returns `{'message': 'Forbidden'}`
        @todo `news` endpoint
    """

    ydc_api_key: Optional[str] = None
    num_web_results: Optional[int] = None
    safesearch: Optional[str] = None
    country: Optional[str] = None
    k: Optional[int] = None
    n_snippets_per_hit: Optional[int] = None
    # @todo deprecate `snippet`, not part of API
    endpoint_type: Literal["search", "news", "rag", "snippet"] = "search"
    # should deprecate n_hits
    n_hits: Optional[int] = None

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        ydc_api_key = get_from_dict_or_env(values, "ydc_api_key", "YDC_API_KEY")
        values["ydc_api_key"] = ydc_api_key

        return values

    def _parse_results(self, raw_search_results: Dict) -> List[Document]:
        """
        Extracts snippets from each hit and puts them in a Document
        Parameters:
            raw_search_results: A dict containing list of hits
        Returns:
            List[YouDocument]: A dictionary of parsed results
        """

        # return news results
        if self.endpoint_type == "news":
            return [
                Document(page_content=result["description"], metadata=result)
                for result in raw_search_results["news"]["results"]
            ]

        docs = []
        for hit in raw_search_results["hits"]:
            n_snippets_per_hit = self.n_snippets_per_hit or len(hit["snippets"])
            for snippet in hit["snippets"][:n_snippets_per_hit]:
                docs.append(
                    Document(
                        page_content=snippet,
                        metadata={
                            "url": hit["url"],
                            "thumbnail_url": hit["thumbnail_url"],
                            "title": hit["title"],
                            "description": hit["description"],
                        },
                    )
                )
                if self.k is not None and len(docs) >= self.k:
                    return docs
        return docs

    def raw_results(
        self,
        query: str,
        **kwargs: Any,
    ) -> Dict:
        """Run query through you.com Search and return hits.

        Args:
            query: The query to search for.
            num_web_results: The maximum number of results to return.
            safesearch: Safesearch settings,
              one of off, moderate, strict, defaults to moderate
            country: Country code
        Returns: YouAPIOutput
        """
        headers = {"X-API-Key": self.ydc_api_key or ""}
        params = {
            "query": query,
            "num_web_results": self.num_web_results,
            "safesearch": self.safesearch,
            "country": self.country,
            **kwargs,
        }

        params = {k: v for k, v in params.items() if v is not None}
        # news endpoint expects `q` instead of `query`
        if self.endpoint_type == "news":
            params["q"] = params["query"]
            del params["query"]

        # @todo deprecate `snippet`, not part of API
        if self.endpoint_type == "snippet":
            self.endpoint_type = "search"
        response = requests.get(
            # type: ignore
            f"{YOU_API_URL}/{self.endpoint_type}",
            params=params,
            headers=headers,
        )
        response.raise_for_status()
        return response.json()

    def results(
        self,
        query: str,
        **kwargs: Any,
    ) -> List[Document]:
        """Run query through you.com Search and parses results into Documents."""

        raw_search_results = self.raw_results(
            query,
            **{key: value for key, value in kwargs.items() if value is not None},
        )
        return self._parse_results(raw_search_results)

    async def raw_results_async(
        self,
        query: str,
        num_web_results: Optional[int] = 5,
        safesearch: Optional[str] = "moderate",
        country: Optional[str] = "US",
    ) -> Dict:
        """Get results from the you.com Search API asynchronously."""

        # Function to perform the API call
        async def fetch() -> str:
            params = {
                "query": query,
                "num_web_results": num_web_results,
                "safesearch": safesearch,
                "country": country,
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{YOU_API_URL}/search", json=params) as res:
                    if res.status == 200:
                        data = await res.text()
                        return data
                    else:
                        raise Exception(f"Error {res.status}: {res.reason}")

        results_json_str = await fetch()
        return json.loads(results_json_str)

    async def results_async(
        self,
        query: str,
        num_web_results: Optional[int] = 5,
        safesearch: Optional[str] = "moderate",
        country: Optional[str] = "US",
    ) -> List[Document]:
        results_json = await self.raw_results_async(
            query=query,
            num_web_results=num_web_results,
            safesearch=safesearch,
            country=country,
        )

        return self._parse_results(results_json["results"])
