"""
Tool implementations for the gaode (https://www.amap.com) weather API.

Documentation: https://lbs.amap.com/api/webservice/summary
API keys:      https://console.amap.com/dev/index
"""

import json
import os
from typing import Any, Type

import requests
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from requests import Response

BASE_URL = "https://restapi.amap.com/v3"  # Gaode web base url


class GaodeWeatherInput(BaseModel):
    """Input for the gaode weather tool."""

    city: str = Field(description="city that require weather query")


class GaodeWeatherTool(BaseTool):
    """Gaode weather query tool"""

    name = "gaode_weather_tool"
    description = "Search for weather information based on the input city"
    args_schema: Type[BaseModel] = GaodeWeatherInput
    return_direct = True

    def _run(self, *args: Any, **kwargs: Any) -> str:
        city = kwargs.get("city")
        if city is None:
            return "Invalid city input!"

        api_key = os.getenv("GAODE_API_KEY")
        if api_key is None:
            return "Please apply for gaode api_key first."

        try:
            city_url = (f"{BASE_URL}/config/district"
                        f"?keywords={city}&subdistrict=0&key={api_key}")
            city_resp = self._get_json_response(city_url)
            if city_resp.get("info") != "OK":
                return "Failed to get city code"

            city_code = city_resp.get("districts")[0].get("adcode")

            weather_url = (f"{BASE_URL}/weather/weatherInfo"
                           f"?city={city_code}&extensions=all&key={api_key}")
            weather_resp = self._get_json_response(weather_url)
            if weather_resp.get("info") != "OK":
                return "Failed to get weather info"

            return json.dumps(weather_resp)

        except Exception as e:
            return f"Failed to get weather info of {city}! error: {str(e)}"

    @staticmethod
    def _get_json_response(url: str) -> Response:
        """Get for gaode json response"""

        with requests.session():
            session = requests.session()
            resp = session.request(
                method="GET",
                url=url,
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
            resp.raise_for_status()
            return resp.json()
