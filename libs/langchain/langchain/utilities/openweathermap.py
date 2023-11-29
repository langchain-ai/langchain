"""Util that calls OpenWeatherMap using PyOWM."""
from typing import Any, Dict, Optional

from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator

from langchain.utils import get_from_dict_or_env


class OpenWeatherMapAPIWrapper(BaseModel):
    """Wrapper for OpenWeatherMap API using PyOWM.

    Docs for using:

    1. Go to OpenWeatherMap and sign up for an API key
    2. Save your API KEY into OPENWEATHERMAP_API_KEY env variable
    3. pip install pyowm
    """

    owm: Any
    openweathermap_api_key: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        openweathermap_api_key = get_from_dict_or_env(
            values, "openweathermap_api_key", "OPENWEATHERMAP_API_KEY"
        )

        try:
            import pyowm

        except ImportError:
            raise ImportError(
                "pyowm is not installed. Please install it with `pip install pyowm`"
            )

        owm = pyowm.OWM(openweathermap_api_key)
        values["owm"] = owm

        return values

    def _format_weather_info(self, location: str, w: Any) -> str:
        detailed_status = w.detailed_status
        wind = w.wind()
        humidity = w.humidity
        temperature = w.temperature("celsius")
        rain = w.rain
        heat_index = w.heat_index
        clouds = w.clouds

        return (
            f"In {location}, the current weather is as follows:\n"
            f"Detailed status: {detailed_status}\n"
            f"Wind speed: {wind['speed']} m/s, direction: {wind['deg']}°\n"
            f"Humidity: {humidity}%\n"
            f"Temperature: \n"
            f"  - Current: {temperature['temp']}°C\n"
            f"  - High: {temperature['temp_max']}°C\n"
            f"  - Low: {temperature['temp_min']}°C\n"
            f"  - Feels like: {temperature['feels_like']}°C\n"
            f"Rain: {rain}\n"
            f"Heat index: {heat_index}\n"
            f"Cloud cover: {clouds}%"
        )

    def run(self, location: str) -> str:
        """Get the current weather information for a specified location."""
        mgr = self.owm.weather_manager()
        observation = mgr.weather_at_place(location)
        w = observation.weather

        return self._format_weather_info(location, w)
