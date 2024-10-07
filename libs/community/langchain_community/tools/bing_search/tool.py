"""Tool for the Bing search API."""

from typing import Dict, List, Literal, Optional, Tuple

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

from langchain_community.utilities.bing_search import BingSearchAPIWrapper


class BingSearchRun(BaseTool):
    """Tool that queries the Bing search API."""

    name: str = "bing_search"
    description: str = (
        "A wrapper around Bing Search. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query."
    )
    api_wrapper: BingSearchAPIWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)


class BingSearchResults(BaseTool):
    """Bing Search tool.

    Setup:
        Install ``langchain-community`` and set environment variable ``BING_SUBSCRIPTION_KEY``.

        .. code-block:: bash

            pip install -U langchain-community
            export BING_SUBSCRIPTION_KEY="your-api-key"

    Instantiation:
        .. code-block:: python

            from langchain_community.tools.bing_search import BingSearchResults
            from langchain_community.utilities import BingSearchAPIWrapper

            api_wrapper = BingSearchAPIWrapper()
            tool = BingSearchResults(api_wrapper=api_wrapper)

    Invocation with args:
        .. code-block:: python

            tool.invoke({"query": "what is the weather in SF?"})

        .. code-block:: python

            "[{'snippet': '<b>San Francisco, CA</b> <b>Weather</b> Forecast, with current conditions, wind, air quality, and what to expect for the next 3 days.', 'title': 'San Francisco, CA Weather Forecast | AccuWeather', 'link': 'https://www.accuweather.com/en/us/san-francisco/94103/weather-forecast/347629'}, {'snippet': 'Tropical Storm Ernesto Forms; Fire <b>Weather</b> Concerns in the Great Basin: Hot Temperatures Return to the South-Central U.S. ... <b>San Francisco CA</b> 37.77°N 122.41°W (Elev. 131 ft) Last Update: 2:21 pm PDT Aug 12, 2024. Forecast Valid: 6pm PDT Aug 12, 2024-6pm PDT Aug 19, 2024 .', 'title': 'National Weather Service', 'link': 'https://forecast.weather.gov/zipcity.php?inputstring=San+Francisco,CA'}, {'snippet': 'Current <b>weather</b> <b>in San Francisco, CA</b>. Check current conditions <b>in San Francisco, CA</b> with radar, hourly, and more.', 'title': 'San Francisco, CA Current Weather | AccuWeather', 'link': 'https://www.accuweather.com/en/us/san-francisco/94103/current-weather/347629'}, {'snippet': 'Everything you need to know about today&#39;s <b>weather</b> <b>in San Francisco, CA</b>. High/Low, Precipitation Chances, Sunrise/Sunset, and today&#39;s Temperature History.', 'title': 'Weather Today for San Francisco, CA | AccuWeather', 'link': 'https://www.accuweather.com/en/us/san-francisco/94103/weather-today/347629'}]"

    Invocation with ToolCall:

        .. code-block:: python

            tool.invoke({"args": {"query":"what is the weather in SF?"}, "id": "1", "name": tool.name, "type": "tool_call"})

        .. code-block:: python

            ToolMessage(
                content="[{'snippet': 'Get the latest <b>weather</b> forecast for <b>San Francisco, CA</b>, including temperature, RealFeel, and chance of precipitation. Find out how the <b>weather</b> will affect your plans and activities in the city of ...', 'title': 'San Francisco, CA Weather Forecast | AccuWeather', 'link': 'https://www.accuweather.com/en/us/san-francisco/94103/weather-forecast/347629'}, {'snippet': 'Radar. Be prepared with the most accurate 10-day forecast for <b>San Francisco, CA</b> with highs, lows, chance of precipitation from The <b>Weather</b> Channel and <b>Weather</b>.com.', 'title': '10-Day Weather Forecast for San Francisco, CA - The Weather Channel', 'link': 'https://weather.com/weather/tenday/l/San+Francisco+CA+USCA0987:1:US'}, {'snippet': 'Tropical Storm Ernesto Forms; Fire <b>Weather</b> Concerns in the Great Basin: Hot Temperatures Return to the South-Central U.S. ... <b>San Francisco CA</b> 37.77°N 122.41°W (Elev. 131 ft) Last Update: 2:21 pm PDT Aug 12, 2024. Forecast Valid: 6pm PDT Aug 12, 2024-6pm PDT Aug 19, 2024 .', 'title': 'National Weather Service', 'link': 'https://forecast.weather.gov/zipcity.php?inputstring=San+Francisco,CA'}, {'snippet': 'Current <b>weather</b> <b>in San Francisco, CA</b>. Check current conditions <b>in San Francisco, CA</b> with radar, hourly, and more.', 'title': 'San Francisco, CA Current Weather | AccuWeather', 'link': 'https://www.accuweather.com/en/us/san-francisco/94103/current-weather/347629'}]",
                artifact=[{'snippet': 'Get the latest <b>weather</b> forecast for <b>San Francisco, CA</b>, including temperature, RealFeel, and chance of precipitation. Find out how the <b>weather</b> will affect your plans and activities in the city of ...', 'title': 'San Francisco, CA Weather Forecast | AccuWeather', 'link': 'https://www.accuweather.com/en/us/san-francisco/94103/weather-forecast/347629'}, {'snippet': 'Radar. Be prepared with the most accurate 10-day forecast for <b>San Francisco, CA</b> with highs, lows, chance of precipitation from The <b>Weather</b> Channel and <b>Weather</b>.com.', 'title': '10-Day Weather Forecast for San Francisco, CA - The Weather Channel', 'link': 'https://weather.com/weather/tenday/l/San+Francisco+CA+USCA0987:1:US'}, {'snippet': 'Tropical Storm Ernesto Forms; Fire <b>Weather</b> Concerns in the Great Basin: Hot Temperatures Return to the South-Central U.S. ... <b>San Francisco CA</b> 37.77°N 122.41°W (Elev. 131 ft) Last Update: 2:21 pm PDT Aug 12, 2024. Forecast Valid: 6pm PDT Aug 12, 2024-6pm PDT Aug 19, 2024 .', 'title': 'National Weather Service', 'link': 'https://forecast.weather.gov/zipcity.php?inputstring=San+Francisco,CA'}, {'snippet': 'Current <b>weather</b> <b>in San Francisco, CA</b>. Check current conditions <b>in San Francisco, CA</b> with radar, hourly, and more.', 'title': 'San Francisco, CA Current Weather | AccuWeather', 'link': 'https://www.accuweather.com/en/us/san-francisco/94103/current-weather/347629'}],
                name='bing_search_results_json',
                tool_call_id='1'
            )

    """  # noqa: E501

    name: str = "bing_search_results_json"
    description: str = (
        "A wrapper around Bing Search. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query. Output is an array of the query results."
    )
    num_results: int = 4
    """Max search results to return, default is 4."""
    api_wrapper: BingSearchAPIWrapper
    response_format: Literal["content_and_artifact"] = "content_and_artifact"

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[str, List[Dict]]:
        """Use the tool."""
        try:
            results = self.api_wrapper.results(query, self.num_results)
            return str(results), results
        except Exception as e:
            return repr(e), []
