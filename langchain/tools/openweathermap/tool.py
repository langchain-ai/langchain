"""Tool for the OpenWeatherMap API."""

from langchain.tools.base import BaseTool
from langchain.utilities import OpenWeatherMapAPIWrapper


class OpenWeatherMapQueryRun(BaseTool):
    """Tool that adds the capability to query using the OpenWeatherMap API."""

    api_wrapper: OpenWeatherMapAPIWrapper

    name = "OpenWeatherMap"
    description = (
        "A wrapper around OpenWeatherMap API. "
        "Useful for fetching current weather information for a specified location. "
        "Input should be a location string (e.g. 'London,GB')."
    )

    def __init__(self) -> None:
        self.api_wrapper = OpenWeatherMapAPIWrapper()
        return

    def _run(self, location: str) -> str:
        """Use the OpenWeatherMap tool."""
        return self.api_wrapper.run(location)

    async def _arun(self, location: str) -> str:
        """Use the OpenWeatherMap tool asynchronously."""
        raise NotImplementedError("OpenWeatherMapQueryRun does not support async")
