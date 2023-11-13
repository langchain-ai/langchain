"""Tool for the Tavily search API."""

from typing import Dict, List, Optional, Type, Union

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.base import BaseTool
from langchain.utilities.tavily_search import TavilySearchAPIWrapper


class TavilyInput(BaseModel):
    query: str = Field(description="search query to look up")


class TavilySearchResults(BaseTool):
    """Tool that queries the Tavily Search API and gets back json."""

    name: str = "tavily_search_results_json"
    description: str = (
        "A search engine optimized for comprehensive, accurate, and trusted results. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query."
    )
    api_wrapper: TavilySearchAPIWrapper
    max_results: int = 5
    args_schema: Type[BaseModel] = TavilyInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Union[List[Dict], str]:
        """Use the tool."""
        try:
            return self.api_wrapper.results(
                query,
                self.max_results,
            )
        except Exception as e:
            return repr(e)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Union[List[Dict], str]:
        """Use the tool asynchronously."""
        try:
            return await self.api_wrapper.results_async(
                query,
                self.max_results,
            )
        except Exception as e:
            return repr(e)


class TavilyAnswer(BaseTool):
    """Tool that queries the Tavily Search API and gets back an answer."""

    name: str = "tavily_answer"
    description: str = (
        "A search engine optimized for comprehensive, accurate, and trusted results. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query. "
        "This returns only the answer - not the original source data."
    )
    api_wrapper: TavilySearchAPIWrapper
    args_schema: Type[BaseModel] = TavilyInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Union[List[Dict], str]:
        """Use the tool."""
        try:
            return self.api_wrapper.raw_results(
                query,
                max_results=5,
                include_answer=True,
                search_depth="basic",
            )["answer"]
        except Exception as e:
            return repr(e)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Union[List[Dict], str]:
        """Use the tool asynchronously."""
        try:
            result = await self.api_wrapper.raw_results_async(
                query,
                max_results=5,
                include_answer=True,
                search_depth="basic",
            )
            return result["answer"]
        except Exception as e:
            return repr(e)
