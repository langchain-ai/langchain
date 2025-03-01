"""Tool for the Tavily search API."""

from typing import Any, Dict, List, Literal, Optional, Type, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field

from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper


class TavilySearchInput(BaseModel):
    """Input for [TavilySearchResults]"""

    query: str = Field(description=("Search query to look up"))
    include_domains: Optional[List[str]] = Field(
        default=[],
        description="A list of domains to specifically include in the search results",
    )
    exclude_domains: Optional[List[str]] = Field(
        default=[],
        description="A list of domains to specifically exclude from the search results",
    )
    search_depth: Optional[Literal["basic", "advanced"]] = Field(
        default="basic",
        description="The depth of the search. It can be 'basic' or 'advanced'",
    )
    include_images: Optional[bool] = Field(
        default=False,
        description="Include a list of query related images in the response",
    )
    time_range: Optional[Literal["day", "week", "month", "year"]] = Field(
        default=None,
        description="The time range back from the current date to filter results",
    )


def _generate_suggestions(params: dict) -> list:
    """Generate helpful suggestions based on the failed search parameters."""
    suggestions = []

    search_depth = params.get("search_depth")
    exclude_domains = params.get("exclude_domains")
    include_domains = params.get("include_domains")
    time_range = params.get("time_range")
    days = params.get("days")

    if time_range:
        suggestions.append("Remove time_range argument")
    elif days:
        suggestions.append("Remove days argument")
    elif include_domains:
        suggestions.append("Remove include_domains argument")
    elif exclude_domains:
        suggestions.append("Remove exclude_domains argument")
    elif search_depth == "basic":
        suggestions.append("Try a more detailed search using 'advanced' search_depth")
    else:
        suggestions.append("Try alternative search terms")

    return suggestions


class TavilySearchResults(BaseTool):  # type: ignore[override, override]
    """Tool that queries the Tavily Search API with dynamically settable parameters."""

    name: str = "tavily_search"
    description: str = (
        "A search engine optimized for comprehensive, accurate, and trusted results. "
        "Useful for when you need to answer questions about current events. "
        "It not only retrieves URLs and snippets, but offers advanced search depths, "
        "domain management, time range filters, and image search, this tool delivers "
        "real-time, accurate, and citation-backed results."
        "Input should be a search query."
    )

    args_schema: Type[BaseModel] = TavilySearchInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 5
    """Max search results to return, 
    
    default is 5
    """
    include_answer: Optional[bool] = False
    """Include a short answer to original query in the search results. 
    
    Default is False.
    """
    include_raw_content: Optional[bool] = False
    """Include cleaned and parsed HTML of each site search results. 
    
    Default is False.
    """
    include_image_descriptions: Optional[bool] = False
    """Include a descriptive text for each image in the search results.
    
    Default is False.
    """
    topic: Optional[Literal["general", "news"]] = "general"
    """The category of the search. Can be "general" or "news".
    
    Default is "general".
    """
    days: Optional[int] = None
    """Number of days back from the current date to include. Only if topic is "news".
    
    Default is None.
    """

    api_wrapper: TavilySearchAPIWrapper

    def __init__(self, **kwargs: Any) -> None:
        # Create api_wrapper with tavily_api_key if provided
        if "tavily_api_key" in kwargs:
            kwargs["api_wrapper"] = TavilySearchAPIWrapper(
                tavily_api_key=kwargs["tavily_api_key"]
            )

        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        include_domains: Optional[List[str]] = [],
        exclude_domains: Optional[List[str]] = [],
        search_depth: Optional[Literal["basic", "advanced"]] = "basic",
        include_images: Optional[bool] = False,
        time_range: Optional[Literal["day", "week", "month", "year"]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Use the tool."""
        try:
            # Execute search with parameters directly
            raw_results = self.api_wrapper.raw_results(
                query=query,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
                search_depth=search_depth,
                include_images=include_images,
                time_range=time_range,
                max_results=self.max_results,
                include_answer=self.include_answer,
                include_raw_content=self.include_raw_content,
                include_image_descriptions=self.include_image_descriptions,
                topic=self.topic,
                days=self.days,
            )

            # Check if results are empty and raise a specific exception
            if not raw_results.get("results", []):
                search_params = {
                    "time_range": time_range,
                    "include_domains": include_domains,
                    "search_depth": search_depth,
                    "exclude_domains": exclude_domains,
                }
                suggestions = _generate_suggestions(search_params)

                # Construct a detailed message for the agent
                error_message = (
                    f"No search results found for '{query}'. "
                    f"Suggestions: {', '.join(suggestions)}. "
                    f"Try modifying your search parameters with one of these approaches."  # noqa: E501
                )
                raise ToolException(error_message)
            return raw_results
        except ToolException:
            # Re-raise tool exceptions
            raise
        except Exception as e:
            raise e

    async def _arun(
        self,
        query: str,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        search_depth: Optional[Literal["basic", "advanced"]] = None,
        include_images: Optional[bool] = None,
        time_range: Optional[Literal["day", "week", "month", "year"]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Use the tool asynchronously."""
        try:
            raw_results = await self.api_wrapper.raw_results_async(
                query=query,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
                search_depth=search_depth,
                include_images=include_images,
                time_range=time_range,
                max_results=self.max_results,
                include_answer=self.include_answer,
                include_raw_content=self.include_raw_content,
                include_image_descriptions=self.include_image_descriptions,
                topic=self.topic,
                days=self.days,
            )

            # Check if results are empty and raise a specific exception
            if not raw_results.get("results", []):
                search_params = {
                    "time_range": time_range,
                    "include_domains": include_domains,
                    "search_depth": search_depth,
                    "exclude_domains": exclude_domains,
                }
                suggestions = _generate_suggestions(search_params)

                # Construct a detailed message for the agent
                error_message = (
                    f"No search results found for '{query}'. "
                    f"Suggestions: {', '.join(suggestions)}. "
                    f"Try modifying your search parameters with one of these approaches."  # noqa: E501
                )
                raise ToolException(error_message)
            return raw_results
        except ToolException:
            # Re-raise tool exceptions
            raise
        except Exception as e:
            # Convert other exceptions to ToolException
            error_message = f"Error during Tavily search: {str(e)}"
            raise ToolException(error_message)


class TavilyAnswer(BaseTool):  # type: ignore[override, override]
    """Tool that queries the Tavily Search API and gets back an answer."""

    name: str = "tavily_answer"
    description: str = (
        "A search engine optimized for comprehensive, accurate, and trusted results. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query. "
        "This returns only the answer - not the original source data."
    )
    api_wrapper: TavilySearchAPIWrapper
    args_schema: Type[BaseModel] = TavilySearchInput

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
