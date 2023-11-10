"""Tool for the DuckDuckGo search API."""

import warnings
from typing import Any, Optional, Type

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.base import BaseTool
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper


class DDGInput(BaseModel):
    query: str = Field(description="search query to look up")


class DuckDuckGoSearchRun(BaseTool):
    """Tool that queries the DuckDuckGo search API."""

    name: str = "duckduckgo_search"
    description: str = (
        "A wrapper around DuckDuckGo Search. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query."
    )
    api_wrapper: DuckDuckGoSearchAPIWrapper = Field(
        default_factory=DuckDuckGoSearchAPIWrapper
    )
    args_schema: Type[BaseModel] = DDGInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)


class DuckDuckGoSearchResults(BaseTool):
    """Tool that queries the DuckDuckGo search API and gets back json."""

    name: str = "DuckDuckGo Results JSON"
    description: str = (
        "A wrapper around Duck Duck Go Search. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query. Output is a JSON array of the query results"
    )
    num_results: int = 4
    api_wrapper: DuckDuckGoSearchAPIWrapper = Field(
        default_factory=DuckDuckGoSearchAPIWrapper
    )
    backend: str = "api"
    args_schema: Type[BaseModel] = DDGInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        res = self.api_wrapper.results(query, self.num_results, backend=self.backend)
        res_strs = [", ".join([f"{k}: {v}" for k, v in d.items()]) for d in res]
        return ", ".join([f"[{rs}]" for rs in res_strs])


def DuckDuckGoSearchTool(*args: Any, **kwargs: Any) -> DuckDuckGoSearchRun:
    """
    Deprecated. Use DuckDuckGoSearchRun instead.

    Args:
        *args:
        **kwargs:

    Returns:
        DuckDuckGoSearchRun
    """
    warnings.warn(
        "DuckDuckGoSearchTool will be deprecated in the future. "
        "Please use DuckDuckGoSearchRun instead.",
        DeprecationWarning,
    )
    return DuckDuckGoSearchRun(*args, **kwargs)
