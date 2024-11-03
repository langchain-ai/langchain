"""Tool for the Oxylabs Search API."""

import json
from typing import Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from langchain_community.utilities.oxylabs_search import OxylabsSearchAPIWrapper


class OxylabsSearchQueryInput(BaseModel):
    """Input for the OxylabsSearch tool."""

    query: str = Field(description="query to retrieve on Oxylabs Search API")
    geo_location: Optional[str] = Field(
        default="California,United States",
        description="Geographic location for the search;"
        " adjust if location-specific information is requested.",
    )


class OxylabsSearchRun(BaseTool):
    """Tool that queries a Oxylabs instance."""

    name: str = "oxylabs_search"
    description: str = (
        "A meta search engine."
        "Ideal for situations where you need to answer questions about current events,"
        "facts, products, recipes, local information, and other topics "
        "that can be explored via web browsing. "
        "The input should be a search query and, if applicable,"
        " a geo_location string to enhance result accuracy. "
        "The output is a compiled, formatted summary of query results. "
    )
    wrapper: OxylabsSearchAPIWrapper
    kwargs: dict = Field(default_factory=dict)
    args_schema: Type[BaseModel] = OxylabsSearchQueryInput

    def _run(
        self,
        query: str,
        geo_location: Optional[str] = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""

        kwargs_ = dict(**self.kwargs)
        kwargs_.update(
            {
                "geo_location": geo_location,
            }
        )

        return self.wrapper.run(query, **kwargs_)

    async def _arun(
        self,
        query: str,
        geo_location: Optional[str] = "",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""

        kwargs_ = dict(**self.kwargs)
        kwargs_.update(
            {
                "geo_location": geo_location,
            }
        )

        return await self.wrapper.arun(query, **kwargs_)


class OxylabsSearchResults(BaseTool):
    """Tool that queries a Oxylabs instance and gets back json."""

    name: str = "oxylabs_search_results"
    description: str = (
        "A meta search engine."
        "Ideal for situations where you need to answer questions about current events,"
        "facts, products, recipes, local information, and other topics "
        "that can be explored via web browsing. "
        "The input should be a search query and, if applicable,"
        " a geo_location string to enhance result accuracy. "
        "The output is a JSON array of response page objects. "
    )
    wrapper: OxylabsSearchAPIWrapper
    kwargs: dict = Field(default_factory=dict)
    args_schema: Type[BaseModel] = OxylabsSearchQueryInput

    model_config = ConfigDict(
        extra="allow",
    )

    def _run(
        self,
        query: str,
        geo_location: Optional[str] = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""

        kwargs_ = dict(**self.kwargs)
        kwargs_.update(
            {
                "geo_location": geo_location,
            }
        )

        return json.dumps(self.wrapper.results(query, **kwargs_))

    async def _arun(
        self,
        query: str,
        geo_location: Optional[str] = "",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""

        kwargs_ = dict(**self.kwargs)
        kwargs_.update(
            {
                "geo_location": geo_location,
            }
        )

        return json.dumps(await self.wrapper.aresults(query, **kwargs_))
