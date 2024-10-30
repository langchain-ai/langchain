"""Tool for the Oxylabs Search API."""

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

import json
from pydantic import BaseModel, ConfigDict, Field
from typing import Type, Optional


from langchain_community.utilities.oxylabs_search import OxylabsSearchAPIWrapper


class OxylabsSearchQueryInput(BaseModel):
    """Input for the OxylabsSearch tool."""

    query: str = Field(description="query to retrieve on Oxylabs Search API")


class OxylabsSearchRun(BaseTool):
    """Tool that queries a Oxylabs instance."""

    name: str = "oxylabs_search"
    description: str = (
        "A meta search engine."
        "Useful for when you need to answer questions about current events."
        "Input should be a search query. Output is a JSON array of the query results"
    )
    wrapper: OxylabsSearchAPIWrapper
    kwargs: dict = Field(default_factory=dict)
    args_schema: Type[BaseModel] = OxylabsSearchQueryInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.wrapper.run(query, **self.kwargs)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        return await self.wrapper.arun(query, **self.kwargs)


class OxylabsSearchResults(BaseTool):
    """Tool that queries a Oxylabs instance and gets back json."""

    name: str = "oxylabs_search_results"
    description: str = (
        "A meta search engine."
        "Useful for when you need to answer questions about current events."
        "Input should be a search query. Output is a JSON array of the query results"
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
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return json.dumps(self.wrapper.results(query, **self.kwargs))

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        return json.dumps(
            await self.wrapper.aresults(query, **self.kwargs)
        )