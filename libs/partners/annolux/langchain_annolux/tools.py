from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field

try:
    from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
    from langchain_core.tools import BaseTool
except ImportError:
    from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun  # type: ignore
    from langchain.tools import BaseTool  # type: ignore

from .utilities import AnnoluxSearchAPIWrapper


class AnnoluxSearchInput(BaseModel):
    """Input for Annolux Search Tool."""

    query: str = Field(description="Search query to look up on the curated web corpus")


class AnnoluxSearchResultsInput(BaseModel):
    """Input for Annolux Search Results Tool."""

    query: str = Field(description="Search query to look up on the curated web corpus")
    max_results: Optional[int] = Field(default=5, description="Maximum number of results to return (1-10)")
    domains: Optional[List[str]] = Field(default=None, description="Optional domain whitelist filters")


class AnnoluxSearchRun(BaseTool):
    """Tool that queries the Annolux Search API and returns text snippets with fetched_at timestamps."""

    name: str = "annolux_search"
    description: str = (
        "A curated English and Chinese search tool designed for AI agents. "
        "Useful for retrieving technical documentation, latest news, and domain-specific knowledge "
        "with explicit snapshot dates to eliminate hallucinations."
    )
    api_wrapper: AnnoluxSearchAPIWrapper = Field(default_factory=AnnoluxSearchAPIWrapper)
    args_schema: Type[BaseModel] = AnnoluxSearchInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        return await self.api_wrapper.arun(query)


class AnnoluxSearchResults(BaseTool):
    """Tool that queries the Annolux Search API and returns structured JSON results with fetched_at timestamps."""

    name: str = "annolux_search_results_json"
    description: str = (
        "A curated search tool that returns structured JSON results (title, url, snippet, fetched_at) "
        "from the Annolux bilingual knowledge index."
    )
    api_wrapper: AnnoluxSearchAPIWrapper = Field(default_factory=AnnoluxSearchAPIWrapper)
    args_schema: Type[BaseModel] = AnnoluxSearchResultsInput

    def _run(
        self,
        query: str,
        max_results: Optional[int] = 5,
        domains: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Dict[str, Any]]:
        """Use the tool."""
        return self.api_wrapper.results(query, max_results=max_results, domains=domains)

    async def _arun(
        self,
        query: str,
        max_results: Optional[int] = 5,
        domains: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> List[Dict[str, Any]]:
        """Use the tool asynchronously."""
        return await self.api_wrapper.results_async(query, max_results=max_results, domains=domains)
