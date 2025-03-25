"""Tool for the OpenWeatherMap API."""

from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import Field

from langchain_community.utilities.openweathermap import OpenWeatherMapAPIWrapper


class OpenWeatherMapQueryRun(BaseTool):  # type: ignore[override]
    """Tool that queries the OpenWeatherMap API."""

    api_wrapper: OpenWeatherMapAPIWrapper = Field(
        default_factory=OpenWeatherMapAPIWrapper  # type: ignore[arg-type]
    )

    name: str = "open_weather_map"
    description: str = (
        "A wrapper around OpenWeatherMap API. "
        "Useful for fetching current weather information for a specified location. "
        "Input should be a location string (e.g. London,GB)."
    )

    def _run(
        self, location: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the OpenWeatherMap tool."""
        return self.api_wrapper.run(location)
