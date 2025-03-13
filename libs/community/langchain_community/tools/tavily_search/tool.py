"""Tool for the Tavily search API."""

from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field

from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper


class TavilyInput(BaseModel):
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
        default="advanced",
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

    if time_range:
        suggestions.append("Remove time_range argument")
    elif include_domains:
        suggestions.append("Remove include_domains argument")
    elif exclude_domains:
        suggestions.append("Remove exclude_domains argument")
    elif search_depth == "advanced":
        suggestions.append("Try a more detailed search using 'advanced' search_depth")
    else:
        suggestions.append("Try alternative search terms")

    return suggestions


class TavilySearchResults(BaseTool):  # type: ignore[override, override]
    """Tool that queries the Tavily Search API and gets back json.

    Setup:
        Install ``langchain-openai`` and ``tavily-python``, and set environment variable ``TAVILY_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-community tavily-python
            export TAVILY_API_KEY="your-api-key"

    Instantiate:

        .. code-block:: python

            from langchain_community.tools import TavilySearchResults

            tool = TavilySearchResults(
                max_results=5,
                topic="general",
                include_answer=False,
                include_raw_content=True,          
                # include_domains=[],
                # exclude_domains=[],
                # search_depth="advanced"
                # include_images=True,
                # include_image_descriptions=True
                # time_range="day",
            )

    Invoke directly with args:

        .. code-block:: python

            tool.invoke({"query": "What happened at the last wimbledon"})

        .. code-block:: json

            {
                'query': 'What happened at the last wimbledon',
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [{'title': "Andy Murray pulls out of the men's singles draw at his last Wimbledon",
                             'url': 'https://www.nbcnews.com/news/sports/andy-murray-wimbledon-tennis-singles-draw-rcna159912',
                             'content': "NBC News Now LONDON — Andy Murray, one of the last decade's most successful ..."
                             'score': 0.6755297,
                             'raw_content': "Andy Murray pulls out of the men's singles draw at his last Wimbledon ....."
                             }],
                'response_time': 2.92
            }

    Invoke with tool call:

        .. code-block:: python

            tool.invoke({"args": {'query': 'who won the last french open'}, "type": "tool_call", "id": "foo", "name": "tavily"})

        .. code-block:: python

            ToolMessage(
                content='{ "url": "https://www.nytimes.com...", "content": "Novak Djokovic won the last French Open by beating Casper Ruud ..." }',
                artifact={
                    'query': 'who won the last french open',
                    'follow_up_questions': None,
                    'answer': 'Novak ...',
                    'images': [
                        'https://www.amny.com/wp-content/uploads/2023/06/AP23162622181176-1200x800.jpg',
                        ...
                        ],
                    'results': [
                        {
                            'title': 'Djokovic ...',
                            'url': 'https://www.nytimes.com...',
                            'content': "Novak...",
                            'score': 0.99505633,
                            'raw_content': 'Tennis\nNovak ...'
                        },
                        ...
                    ],
                    'response_time': 2.92
                },
                tool_call_id='1',
                name='tavily_search_results_json',
            )

    """  # noqa: E501

    name: str = "tavily_search_results_json"
    description: str = (
        "A search engine optimized for comprehensive, accurate, and trusted results. "
        "Useful for when you need to answer questions about current events. "
        "It not only retrieves URLs and snippets, but offers advanced search depths, "
        "domain management, time range filters, and image search, this tool delivers "
        "real-time, accurate, and citation-backed results."
        "Input should be a search query."
    )

    args_schema: Type[BaseModel] = TavilyInput
    handle_tool_error: bool = True

    include_domains: Optional[List[str]] = None
    """A list of domains to specifically include in the search results

    default is None
    """
    exclude_domains: Optional[List[str]] = None
    """A list of domains to specifically exclude from the search results

    default is None
    """
    search_depth: Optional[Literal["basic", "advanced"]] = "advanced"
    """The depth of the search. It can be "basic" or "advanced"
    
    default is "advanced"
    """
    include_images: Optional[bool] = False
    """Include a list of query related images in the response
    
    default is False
    """
    time_range: Optional[Literal["day", "week", "month", "year"]] = None
    """The time range back from the current date to filter results
    
    default is None
    """
    max_results: Optional[int] = 5
    """Max search results to return, 
    
    default is 5
    """
    topic: Optional[Literal["general", "news"]] = "general"
    """The category of the search. Can be "general" or "news".
    
    Default is "general".
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
    api_wrapper: TavilySearchAPIWrapper = Field(default_factory=TavilySearchAPIWrapper)  # type: ignore[arg-type]
    response_format: Literal["content_and_artifact"] = "content_and_artifact"

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
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        search_depth: Optional[Literal["basic", "advanced"]] = "advanced",
        include_images: Optional[bool] = False,
        time_range: Optional[Literal["day", "week", "month", "year"]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Union[List[Dict[str, str]], str], Dict]:
        """Use the tool."""
        try:
            # Execute search with parameters directly
            raw_results = self.api_wrapper.raw_results(
                query=query,
                include_domains=include_domains
                if include_domains
                else self.include_domains,
                exclude_domains=exclude_domains
                if exclude_domains
                else self.exclude_domains,
                search_depth=search_depth if search_depth else self.search_depth,
                include_images=include_images
                if include_images
                else self.include_images,
                time_range=time_range if time_range else self.time_range,
                max_results=self.max_results,
                include_answer=self.include_answer,
                include_raw_content=self.include_raw_content,
                include_image_descriptions=self.include_image_descriptions,
                topic=self.topic,
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
            return self.api_wrapper.clean_results(raw_results["results"]), raw_results
        except ToolException:
            # Re-raise tool exceptions
            raise
        except Exception as e:
            return repr(e), {}

    async def _arun(
        self,
        query: str,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        search_depth: Optional[Literal["basic", "advanced"]] = "advanced",
        include_images: Optional[bool] = False,
        time_range: Optional[Literal["day", "week", "month", "year"]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Union[List[Dict[str, str]], str], Dict]:
        """Use the tool asynchronously."""
        try:
            raw_results = await self.api_wrapper.raw_results_async(
                query=query,
                include_domains=include_domains
                if include_domains
                else self.include_domains,
                exclude_domains=exclude_domains
                if exclude_domains
                else self.exclude_domains,
                search_depth=search_depth if search_depth else self.search_depth,
                include_images=include_images
                if include_images
                else self.include_images,
                time_range=time_range if time_range else self.time_range,
                max_results=self.max_results,
                include_answer=self.include_answer,
                include_raw_content=self.include_raw_content,
                include_image_descriptions=self.include_image_descriptions,
                topic=self.topic,
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
            return self.api_wrapper.clean_results(raw_results["results"]), raw_results
        except ToolException:
            # Re-raise tool exceptions
            raise
        except Exception as e:
            return repr(e), {}


class TavilyAnswer(BaseTool):  # type: ignore[override, override]
    """Tool that queries the Tavily Search API and gets back an answer."""

    name: str = "tavily_answer"
    description: str = (
        "A search engine optimized for comprehensive, accurate, and trusted results. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query. "
        "This returns only the answer - not the original source data."
    )
    api_wrapper: TavilySearchAPIWrapper = Field(default_factory=TavilySearchAPIWrapper)  # type: ignore[arg-type]
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
