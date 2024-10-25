"""
Tool for the AskNews API.

To use this tool, you must first set your credentials as environment variables:
    ASKNEWS_CLIENT_ID
    ASKNEWS_CLIENT_SECRET
"""

from typing import Any, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.asknews import AskNewsAPIWrapper


class SearchInput(BaseModel):
    """Input for the AskNews Search tool."""

    query: str = Field(
        description="Search query to be used for finding real-time or historical news "
        "information."
    )
    hours_back: Optional[int] = Field(
        0,
        description="If the Assistant deems that the event may have occurred more "
        "than 48 hours ago, it estimates the number of hours back to search. For "
        "example, if the event was one month ago, the Assistant may set this to 720. "
        "One week would be 168. The Assistant can estimate up to on year back (8760).",
    )


class AskNewsSearch(BaseTool):  # type: ignore[override]
    """Tool that searches the AskNews API."""

    name: str = "asknews_search"
    description: str = (
        "This tool allows you to perform a search on up-to-date news and historical "
        "news. If you needs news from more than 48 hours ago, you can estimate the "
        "number of hours back to search."
    )
    api_wrapper: AskNewsAPIWrapper = Field(default_factory=AskNewsAPIWrapper)  # type: ignore[arg-type]
    max_results: int = 10
    args_schema: Optional[Type[BaseModel]] = SearchInput

    def _run(
        self,
        query: str,
        hours_back: int = 0,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> str:
        """Use the tool."""
        try:
            return self.api_wrapper.search_news(
                query,
                hours_back=hours_back,
                max_results=self.max_results,
            )
        except Exception as e:
            return repr(e)

    async def _arun(
        self,
        query: str,
        hours_back: int = 0,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> str:
        """Use the tool asynchronously."""
        try:
            return await self.api_wrapper.asearch_news(
                query,
                hours_back=hours_back,
                max_results=self.max_results,
            )
        except Exception as e:
            return repr(e)
