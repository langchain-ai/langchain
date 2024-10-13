"""
Tool implementations for the gaode (https://www.amap.com) weather API.

Documentation: https://lbs.amap.com/api/webservice/summary
API keys:      https://console.amap.com/dev/index
"""

import json
from typing import Any, Type

import requests
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

CITY_CODE_URL = "https://restapi.amap.com/v3/config/district"
WEATHER_URL = "https://restapi.amap.com/v3/weather/weatherInfo"


class GaodeWeatherInput(BaseModel):
    """Input for the gaode weather tool."""

    city: str = Field(description="city that require weather query")


class GaodeWeatherTool(BaseTool):
    """Gaode weather query tool"""

    name = "gaode_weather_tool"
    description = "Query for weather information based on the input city"
    args_schema: Type[BaseModel] = GaodeWeatherInput
    return_direct = True
    api_key: str

    def _run(self, *args: Any, **kwargs: Any) -> str:
        api_key = self.api_key
        if api_key is None:
            raise ValueError("Please set GAODE_API_KEY environment variable")

        city = kwargs.get("city")
        if city is None:
            raise ValueError("Invalid city input!")

        try:
            city_url = f"{CITY_CODE_URL}?keywords={city}&subdistrict=0&key={api_key}"

            with requests.session():
                session = requests.session()
                city_resp = session.request(
                    method="GET",
                    url=city_url,
                    headers={"Content-Type": "application/json; charset=utf-8"},
                )
                city_data = city_resp.json()
                if city_data.get("info") != "OK":
                    return "Failed to get city code"

                city_code = city_data.get("districts")[0].get("adcode")
                weather_url = (
                    f"{WEATHER_URL}?city={city_code}&extensions=all&key={api_key}"
                )
                weather_resp = session.request(
                    method="GET",
                    url=weather_url,
                    headers={"Content-Type": "application/json; charset=utf-8"},
                )
                weather_data = weather_resp.json()
                if weather_data.get("info") != "OK":
                    return "Failed to get weather info"

                return json.dumps(weather_data)

        except Exception as e:
            return f"Failed to get weather info of {city}! error: {str(e)}"
