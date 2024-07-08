"""Util that calls you.com Search API.

In order to set this up, follow instructions at:
https://documentation.you.com/quickstart
"""

import warnings
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
    """Output from you.com API."""

    hits: List[YouHit] = Field(
        description="A list of dictionaries containing the results"
    )


class YouDocument(BaseModel):
    """Output of parsing one snippet."""

    page_content: str = Field(description="One snippet of text")
    metadata: YouHitMetadata


class YouSearchAPIWrapper(BaseModel):
    """Wrapper for you.com Search and News API.

    To connect to the You.com api requires an API key which
    you can get at https://api.you.com.
    You can check out the docs at https://documentation.you.com/api-reference/.

    You need to set the environment variable `YDC_API_KEY` for retriever to operate.

    Attributes
    ----------
    ydc_api_key: str, optional
        you.com api key, if YDC_API_KEY is not set in the environment
    endpoint_type: str, optional
        you.com endpoints: search, news, rag;
        `web` and `snippet` alias `search`
        `rag` returns `{'message': 'Forbidden'}`
        @todo `news` endpoint
    num_web_results: int, optional
        The max number of web results to return, must be under 20.
        This is mapped to the `count` query parameter for the News API.
    safesearch: str, optional
        Safesearch settings, one of off, moderate, strict, defaults to moderate
    country: str, optional
        Country code, ex: 'US' for United States, see api docs for list
    search_lang: str, optional
        (News API) Language codes, ex: 'en' for English, see api docs for list
    ui_lang: str, optional
        (News API) User interface language for the response, ex: 'en' for English,
                   see api docs for list
    spellcheck: bool, optional
        (News API) Whether to spell check query or not, defaults to True
    k: int, optional
        max number of Documents to return using `results()`
    n_hits: int, optional, deprecated
        Alias for num_web_results
    n_snippets_per_hit: int, optional
        limit the number of snippets returned per hit
    """

    ydc_api_key: Optional[str] = None

    # @todo deprecate `snippet`, not part of API
    endpoint_type: Literal["search", "news", "rag", "snippet"] = "search"

    # Common fields between Search and News API
    num_web_results: Optional[int] = None
    safesearch: Optional[Literal["off", "moderate", "strict"]] = None
    country: Optional[str] = None

    # News API specific fields
    search_lang: Optional[str] = None
    ui_lang: Optional[str] = None
    spellcheck: Optional[bool] = None

    k: Optional[int] = None
    n_snippets_per_hit: Optional[int] = None
    # should deprecate n_hits
    n_hits: Optional[int] = None

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        ydc_api_key = get_from_dict_or_env(values, "ydc_api_key", "YDC_API_KEY")
        values["ydc_api_key"] = ydc_api_key

        return values

    @root_validator
    def warn_if_set_fields_have_no_effect(cls, values: Dict) -> Dict:
        if values["endpoint_type"] != "news":
            news_api_fields = ("search_lang", "ui_lang", "spellcheck")
            for field in news_api_fields:
                if values[field]:
                    warnings.warn(
                        (
                            f"News API-specific field '{field}' is set but "
                            f"`endpoint_type=\"{values['endpoint_type']}\"`. "
                            "This will have no effect."
                        ),
                        UserWarning,
                    )
        if values["endpoint_type"] not in ("search", "snippet"):
            if values["n_snippets_per_hit"]:
                warnings.warn(
                    (
                        "Field 'n_snippets_per_hit' only has effect on "
                        '`endpoint_type="search"`.'
                    ),
                    UserWarning,
                )
        return values

    @root_validator
    def warn_if_deprecated_endpoints_are_used(cls, values: Dict) -> Dict:
        if values["endpoint_type"] == "snippets":
            warnings.warn(
                (
                    f"`endpoint_type=\"{values['endpoint_type']}\"` is deprecated. "
                    'Use `endpoint_type="search"` instead.'
                ),
                DeprecationWarning,
            )
        return values

    def _generate_params(self, query: str, **kwargs: Any) -> Dict:
        """
        Parse parameters required for different You.com APIs.

        Args:
            query: The query to search for.
        """
        params = {
            "safesearch": self.safesearch,
            "country": self.country,
            **kwargs,
        }

        # Add endpoint-specific params
        if self.endpoint_type in ("search", "snippet"):
            params.update(
                query=query,
                num_web_results=self.num_web_results,
            )
        elif self.endpoint_type == "news":
            params.update(
                q=query,
                count=self.num_web_results,
                search_lang=self.search_lang,
                ui_lang=self.ui_lang,
                spellcheck=self.spellcheck,
            )

        params = {k: v for k, v in params.items() if v is not None}
        return params

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
            news_results = raw_search_results["news"]["results"]
            if self.k is not None:
                news_results = news_results[: self.k]
            return [
                Document(page_content=result["description"], metadata=result)
                for result in news_results
            ]

        docs = []
        for hit in raw_search_results["hits"]:
            n_snippets_per_hit = self.n_snippets_per_hit or len(hit.get("snippets"))
            for snippet in hit.get("snippets")[:n_snippets_per_hit]:
                docs.append(
                    Document(
                        page_content=snippet,
                        metadata={
                            "url": hit.get("url"),
                            "thumbnail_url": hit.get("thumbnail_url"),
                            "title": hit.get("title"),
                            "description": hit.get("description"),
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
        Returns: YouAPIOutput
        """
        headers = {"X-API-Key": self.ydc_api_key or ""}
        params = self._generate_params(query, **kwargs)

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
        **kwargs: Any,
    ) -> Dict:
        """Get results from the you.com Search API asynchronously."""

        headers = {"X-API-Key": self.ydc_api_key or ""}
        params = self._generate_params(query, **kwargs)

        # @todo deprecate `snippet`, not part of API
        if self.endpoint_type == "snippet":
            self.endpoint_type = "search"

        async with aiohttp.ClientSession() as session:
            async with session.get(
                url=f"{YOU_API_URL}/{self.endpoint_type}",
                params=params,
                headers=headers,
            ) as res:
                if res.status == 200:
                    results = await res.json()
                    return results
                else:
                    raise Exception(f"Error {res.status}: {res.reason}")

    async def results_async(
        self,
        query: str,
        **kwargs: Any,
    ) -> List[Document]:
        raw_search_results_async = await self.raw_results_async(
            query,
            **{key: value for key, value in kwargs.items() if value is not None},
        )
        return self._parse_results(raw_search_results_async)
