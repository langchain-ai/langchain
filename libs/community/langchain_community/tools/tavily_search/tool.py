"""Tool for the Tavily search API."""

from typing import Any, Dict, List, Literal, Optional, Type, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field, create_model

from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper


class TavilySearchInput(BaseModel):
    """
    Input for [TavilyAnswer]
    TavilyAnswer returns an answer to the search query and only allows for a query input
    NOTE: DynamicTavilyInput is the input for [TavilySearchResults]
    """

    query: str = Field(description="search query to look up")


def _generate_suggestions(params: dict) -> list:
    """Generate helpful suggestions based on the failed search parameters."""
    suggestions = []

    search_depth = params.get("search_depth")
    topic = params.get("topic")
    include_domains = params.get("include_domains")
    time_range = params.get("time_range")
    days = params.get("days")

    if topic == "news":
        suggestions.append("Try a more general search using 'general' topic")
    elif time_range:
        suggestions.append(f"Remove time_range argument")
    elif days:
        suggestions.append(f"Remove days argument")
    elif include_domains:
        suggestions.append("Remove include_domains argument")
    elif search_depth == "basic":
        suggestions.append("Try a more detailed search using 'advanced' search_depth")
    elif topic == "news":
        suggestions.append("Try a more general search using 'general' topic")
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

    handle_tool_error: bool = True

    # Default parameters
    max_results: Optional[int] = 5
    """Max search results to return, default is 5"""
    search_depth: Optional[Literal["basic", "advanced"]] = "advanced"
    """The depth of the search. It can be "basic" or "advanced"
    
    .. versionadded:: 0.2.5
    """
    include_domains: Optional[List[str]] = []
    """A list of domains to specifically include in the search results. 
    
    Default is None, which includes all domains.
    """
    exclude_domains: Optional[List[str]] = []
    """A list of domains to specifically exclude from the search results. 
    
    Default is None, which doesn't exclude any domains.
    """
    include_answer: Optional[bool] = False
    """Include a short answer to original query in the search results. 
    
    Default is False.
    """
    include_raw_content: Optional[bool] = False
    """Include cleaned and parsed HTML of each site search results. 
    
    Default is False.
    """
    include_images: Optional[bool] = False
    """Include a list of query related images in the response. 
    
    Default is False.
    """
    include_image_descriptions: Optional[bool] = False
    """When include_images is True, also add a descriptive text for each image.
    
    Default is False.
    """
    topic: Optional[Literal["general", "news"]] = "general"
    """The category of the search. Can be "general" or "news".
    
    Default is "general".
    """
    time_range: Optional[
        Literal["day", "week", "month", "year", "d", "w", "m", "y"]
    ] = None
    """The time range back from the current date to filter results.
    
    Can be "day", "week", "month", "year", "d", "w", "m", "y".
    Default is None.
    """
    days: Optional[int] = None
    """Number of days back from the current date to include. Only if topic is "news".
    
    Default is None.
    """

    # Flags to control which parameters are settable by the agent
    settable_max_results: bool = False
    settable_search_depth: bool = False
    settable_include_domains: bool = False
    settable_exclude_domains: bool = False
    settable_include_answer: bool = False
    settable_include_raw_content: bool = False
    settable_include_images: bool = False
    settable_include_image_descriptions: bool = False
    settable_topic: bool = False
    settable_time_range: bool = False
    settable_days: bool = False

    api_wrapper: TavilySearchAPIWrapper

    def __init__(self, **kwargs: Any) -> None:
        # Extract which parameters should be settable
        settable_params = {}
        for param in [
            "max_results",
            "search_depth",
            "include_domains",
            "exclude_domains",
            "include_answer",
            "include_raw_content",
            "include_images",
            "include_image_descriptions",
            "topic",
            "time_range",
            "days",
        ]:
            settable_key = f"settable_{param}"
            if settable_key in kwargs:
                settable_params[settable_key] = kwargs[settable_key]

        # Create api_wrapper with tavily_api_key if provided
        if "tavily_api_key" in kwargs:
            kwargs["api_wrapper"] = TavilySearchAPIWrapper(
                tavily_api_key=kwargs["tavily_api_key"]
            )

        # Create dynamic schema
        fields: Dict[str, Any] = {
            "query": (str, Field(description="Search query to look up"))
        }

        # Add fields that are settable
        param_descriptions = {
            "max_results": "Maximum number of search results to return",
            "search_depth": "Search depth, must be either 'basic' or 'advanced'",
            "include_domains": "List of domains to limit the search results to",
            "exclude_domains": "List of domains to exclude from the search results",
            "include_answer": "Whether to include an AI-generated answer in the response",  # noqa: E501
            "include_raw_content": "Whether to include raw content in the search results",  # noqa: E501
            "include_images": "Whether to include images in the search results",
            "include_image_descriptions": "Whether to include image descriptions in the search results",  # noqa: E501
            "topic": "Topic to focus the search on, must be either 'general' or 'news'",
            "time_range": "Time range of the search, must be 'day', 'week', 'month', 'year'",  # noqa: E501
            "days": "Number of days to look back in search, only used if topic is 'news'",  # noqa: E501
        }

        param_types = {
            "max_results": int,
            "search_depth": Literal["basic", "advanced"],
            "include_domains": List[str],
            "exclude_domains": List[str],
            "include_answer": bool,
            "include_raw_content": bool,
            "include_images": bool,
            "include_image_descriptions": bool,
            "topic": Literal["general", "news"],
            "time_range": Literal["day", "week", "month", "year", "d", "w", "m", "y"],
            "days": int,
        }

        for param, desc in param_descriptions.items():
            settable_key = f"settable_{param}"
            if settable_key in settable_params and settable_params[settable_key]:
                fields[param] = (Optional[param_types[param]], Field(description=desc))

        # Create the dynamic model
        DynamicTavilyInput = create_model("DynamicTavilyInput", **fields)  # type: ignore[call-overload]
        kwargs["args_schema"] = DynamicTavilyInput

        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        time_range: Optional[
            Literal["day", "week", "month", "year", "d", "w", "m", "y"]
        ] = None,
        include_domains: Optional[List[str]] = None,
        max_results: Optional[int] = None,
        search_depth: Optional[Literal["basic", "advanced"]] = None,
        exclude_domains: Optional[List[str]] = None,
        include_answer: Optional[bool] = None,
        include_raw_content: Optional[bool] = None,
        include_images: Optional[bool] = None,
        include_image_descriptions: Optional[bool] = None,
        topic: Optional[Literal["general", "news"]] = None,
        days: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Use the tool."""
        try:
            # Execute search with parameters directly
            raw_results = self.api_wrapper.raw_results(
                query=query,
                time_range=time_range if time_range else self.time_range,
                include_domains=include_domains
                if include_domains
                else self.include_domains,
                max_results=max_results if max_results else self.max_results,
                search_depth=search_depth if search_depth else self.search_depth,
                exclude_domains=exclude_domains
                if exclude_domains
                else self.exclude_domains,
                include_answer=include_answer
                if include_answer
                else self.include_answer,
                include_raw_content=include_raw_content
                if include_raw_content
                else self.include_raw_content,
                include_images=include_images
                if include_images
                else self.include_images,
                include_image_descriptions=include_image_descriptions
                if include_image_descriptions
                else self.include_image_descriptions,
                topic=topic if topic else self.topic,
                days=days if days else self.days,
            )

            # Check if results are empty and raise a specific exception
            if "results" not in raw_results or not raw_results["results"]:
                search_params = {
                    "time_range": time_range if time_range else self.time_range,
                    "days": days if days else self.days,
                    "include_domains": include_domains
                    if include_domains
                    else self.include_domains,
                    "topic": topic if topic else self.topic,
                    "search_depth": search_depth if search_depth else self.search_depth,
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
        time_range: Optional[
            Literal["day", "week", "month", "year", "d", "w", "m", "y"]
        ] = None,
        include_domains: Optional[List[str]] = None,
        max_results: Optional[int] = None,
        search_depth: Optional[Literal["basic", "advanced"]] = None,
        exclude_domains: Optional[List[str]] = None,
        include_answer: Optional[bool] = None,
        include_raw_content: Optional[bool] = None,
        include_images: Optional[bool] = None,
        include_image_descriptions: Optional[bool] = None,
        topic: Optional[Literal["general", "news"]] = None,
        days: Optional[int] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Use the tool asynchronously."""
        try:
            raw_results = await self.api_wrapper.raw_results_async(
                query=query,
                time_range=time_range if time_range else self.time_range,
                include_domains=include_domains
                if include_domains
                else self.include_domains,
                max_results=max_results if max_results else self.max_results,
                search_depth=search_depth if search_depth else self.search_depth,
                exclude_domains=exclude_domains
                if exclude_domains
                else self.exclude_domains,
                include_answer=include_answer
                if include_answer
                else self.include_answer,
                include_raw_content=include_raw_content
                if include_raw_content
                else self.include_raw_content,
                include_images=include_images
                if include_images
                else self.include_images,
                include_image_descriptions=include_image_descriptions
                if include_image_descriptions
                else self.include_image_descriptions,
                topic=topic if topic else self.topic,
                days=days if days else self.days,
            )

            # Check if results are empty and raise a specific exception
            results = raw_results.get("results", [])
            if not results:
                search_params = {
                    "time_range": time_range if time_range else self.time_range,
                    "days": days if days else self.days,
                    "include_domains": include_domains
                    if include_domains
                    else self.include_domains,
                    "topic": topic if topic else self.topic,
                    "search_depth": search_depth if search_depth else self.search_depth,
                }
                suggestions = _generate_suggestions(search_params)
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
