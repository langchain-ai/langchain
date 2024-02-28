"""Tool for the Google Maps and National Weather Service APIs."""

from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool

from langchain_community.utilities.google_weatherservice import GoogleNationalWeatherAPI


class GoogleNationalWeatherServiceQueryRun(BaseTool):
    """Tool that queries the OpenWeatherMap API."""

    api_wrapper: GoogleNationalWeatherAPI = Field(
        default_factory=GoogleNationalWeatherAPI
    )

    name: str = "google_national_weather_service"
    description: str = (
        "A wrapper around the Google Maps and National Weather Service APIs. "
        "Useful for fetching current weather information for a specified location. "
        "Input should be a location string (e.g. Chicago)."
    )

    def _run(
        self, location: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the GoogleNationalWeather tool."""
        return self.api_wrapper.run(location)