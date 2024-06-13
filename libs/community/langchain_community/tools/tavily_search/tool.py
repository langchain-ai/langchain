"""Tool for the Tavily search API."""

from typing import Dict, List, Optional, Type, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper


class TavilyInput(BaseModel):
    """Input for the Tavily tool."""

    query: str = Field(description="search query to look up")


class TavilySearchResults(BaseTool):
    """Tool that queries the Tavily Search API and gets back json.

    Setup:
        Install ``langchain-openai`` and ``tavily-python``, and set environment variable ``TAVILY_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-openai
            export TAVILY_API_KEY="your-api-key"

    Instantiate:

        .. code-block:: python

            from langchain_community.tools.tavily_search import TavilySearchResults

            tool = TavilySearchResults(
                # max_results= 5
                # search_depth = "advanced"
                # include_domains = []
                # exclude_domains = []
                # include_answer = False
                # include_raw_content = False
                # include_images = False
            )

    Invoke:

        .. code-block:: python

            tool = TavilySearchResults(max_results=3)
            tool.invoke("What is the weather?")

        .. code-block:: python

            [{'url': 'https://www.weatherapi.com/',
            'content': "{'location': {'name': 'Current', 'region': 'Harbour Island', 'country': 'Bahamas', 'lat': 25.43, 'lon': -76.78, 'tz_id': 'America/Nassau', 'localtime_epoch': 1718077801, 'localtime': '2024-06-10 23:50'}, 'current': {'last_updated_epoch': 1718077500, 'last_updated': '2024-06-10 23:45', 'temp_c': 27.9, 'temp_f': 82.1, 'is_day': 0, 'condition': {'text': 'Patchy rain nearby', 'icon': '//cdn.weatherapi.com/weather/64x64/night/176.png', 'code': 1063}, 'wind_mph': 14.5, 'wind_kph': 23.4, 'wind_degree': 161, 'wind_dir': 'SSE', 'pressure_mb': 1014.0, 'pressure_in': 29.94, 'precip_mm': 0.01, 'precip_in': 0.0, 'humidity': 88, 'cloud': 74, 'feelslike_c': 33.1, 'feelslike_f': 91.5, 'windchill_c': 27.9, 'windchill_f': 82.1, 'heatindex_c': 33.1, 'heatindex_f': 91.5, 'dewpoint_c': 25.6, 'dewpoint_f': 78.1, 'vis_km': 10.0, 'vis_miles': 6.0, 'uv': 1.0, 'gust_mph': 20.4, 'gust_kph': 32.9}}"},
            {'url': 'https://www.localconditions.com/weather-ninnescah-kansas/67069/',
            'content': 'The following chart reports what the hourly Ninnescah, KS temperature has been today, from 12:56 AM to 3:56 AM Tue, May 21st 2024. The lowest temperature reading has been 73.04 degrees fahrenheit at 3:56 AM, while the highest temperature is 75.92 degrees fahrenheit at 12:56 AM. Ninnescah KS detailed current weather report for 67069 in Kansas.'},
            {'url': 'https://www.weather.gov/forecastmaps/',
            'content': 'Short Range Forecasts. Short range forecast products depicting pressure patterns, circulation centers and fronts, and types and extent of precipitation. 12 Hour | 24 Hour | 36 Hour | 48 Hour.'}]

        When converting ``TavilySearchResults`` to a tool, you may want to not return all of the content resulting from ``invoke``. You can select what parts of the response to keep depending on your use case.

    """  # noqa: E501

    name: str = "tavily_search_results_json"
    description: str = (
        "A search engine optimized for comprehensive, accurate, and trusted results. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query."
    )
    api_wrapper: TavilySearchAPIWrapper = Field(default_factory=TavilySearchAPIWrapper)  # type: ignore[arg-type]
    max_results: int = 5
    """Max search results to return, default is 5"""
    search_depth: str = "advanced"
    '''The depth of the search. It can be "basic" or "advanced"'''
    include_domains: List[str] = []
    """A list of domains to specifically include in the search results. Default is None, which includes all domains."""  # noqa: E501
    exclude_domains: List[str] = []
    """A list of domains to specifically exclude from the search results. Default is None, which doesn't exclude any domains."""  # noqa: E501
    include_answer: bool = False
    """Include a short answer to original query in the search results. Default is False."""  # noqa: E501
    include_raw_content: bool = False
    """Include cleaned and parsed HTML of each site search results. Default is False."""
    include_images: bool = False
    """Include a list of query related images in the response. Default is False."""
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
                self.search_depth,
                self.include_domains,
                self.exclude_domains,
                self.include_answer,
                self.include_raw_content,
                self.include_images,
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
                self.search_depth,
                self.include_domains,
                self.exclude_domains,
                self.include_answer,
                self.include_raw_content,
                self.include_images,
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
