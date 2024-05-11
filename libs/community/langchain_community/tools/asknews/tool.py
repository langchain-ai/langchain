"""
Tool for the AskNews API.

To use this tool, you must first set your credentials as environment variables:
    ASKNEWS_CLIENT_ID
    ASKNEWS_CLIENT_SECRET
"""

from typing import Literal, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from langchain_community.utilities.asknews import AskNewsAPIWrapper


class SearchInput(BaseModel):
    """Input for the AskNews Search tool."""

    query: str = Field(description="Search query to look up")
    method: Optional[Literal["nl", "kw"]] = Field(
        "nl",
        description="Method to use for search. 'nl' for natural language, "
        "'kw' for keywords.",
    )
    historical: Optional[bool] = Field(False, description="Search for historical news.")


class AskNewsSearch(BaseTool):
    """Tool that searches the AskNews API."""

    name: str = "asknews_search"
    description: str = (
        "This tool allows you perform search on news in a natural language or by "
        "using keywords. Useful for when you need to answer questions about current"
        "events or news. Input should be a search query in a natural language if "
        "method is specifed as 'nl' or keywords if method is specified as 'kw'."
    )
    api_wrapper: AskNewsAPIWrapper = Field(default_factory=AskNewsAPIWrapper)
    max_results: int = 10
    args_schema: Type[BaseModel] = SearchInput

    def _run(
        self,
        query: str,
        method: Literal["nl", "kw"] = "nl",
        historical: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:
            return self.api_wrapper.search_news(
                query,
                method=method,
                historical=historical,
                max_results=self.max_results,
            )
        except Exception as e:
            return repr(e)

    async def _arun(
        self,
        query: str,
        method: Literal["nl", "kw"] = "nl",
        historical: bool = False,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        try:
            return await self.api_wrapper.asearch_news(
                query,
                method=method,
                historical=historical,
                max_results=self.max_results,
            )
        except Exception as e:
            return repr(e)
