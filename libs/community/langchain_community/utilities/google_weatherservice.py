import os
from typing import Any, Dict, Optional, Type

import requests
from langchain.pydantic_v1 import BaseModel, Field, root_validator
from langchain.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env


class GCPWeatherInput(BaseModel):
    location: str = Field(
        description="""The location to look up. Can be a standalone city or town,
         or could include the accompanying state"""
    )


class GoogleNationalWeatherAPI(BaseTool):
    name = "Temperature Query Tool"
    description = "Use this tool to return the temperature for a location."
    args_schema: Type[BaseModel] = GCPWeatherInput

    gmaps: Any
    google_api_key: Optional[str] = None

    @root_validator(pre=True)
    # Check the following link for updating the root_validator decorator; 
    # pull from pydantic v2, instead of langchain/pydantic v1. 
    # https://docs.pydantic.dev/latest/api/functional_validators/
    # #pydantic.functional_validators.model_validator
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        google_api_key = get_from_dict_or_env(
            values, "google_api_key", "GOOGLE_API_KEY"
        )

        try:
            import googlemaps

        except ImportError:
            raise ImportError(
                """googlemaps is not installed. 
                Please install it with `pip install googlemaps`"""
            )

        gmaps = googlemaps.Client(google_api_key)
        values["gmaps"] = gmaps

        return values

    def _geocode(
        self,
        location: str = Field(
            description="The stated town or city, with the associated state, if stated"
        ),
    ) -> dict:
        import googlemaps

        gmaps = googlemaps.Client(key=os.environ["GOOGLE_API_KEY"])
        geocode_result = gmaps.geocode(location)

        return geocode_result[0].get("geometry").get("location").get(
            "lat"
        ), geocode_result[0].get("geometry").get("location").get("lng")

    def _return_temperature(
        self,
        latitude: float = Field(description="Latitude for the location"),
        longitude: float = Field(description="Longitude for the location"),
    ):
        weather = requests.get(f"https://api.weather.gov/points/{latitude},{longitude}")
        forecast_hourly = requests.get(
            weather.json().get("properties").get("forecastHourly")
        )
        temp = forecast_hourly.json()["properties"]["periods"][0]["temperature"]

        return f"Chicago currently has a temperature of {temp} Fahrenheit"

    def _run(
        self,
        location: str = Field(
            description="The stated town or city, with the associated state, if stated"
        ),
    ) -> str:
        """
        This method is used to return the temperature at a given location
        """

        latitude, longitude = self._geocode(location)
        temp = self._return_temperature(latitude, longitude)

        return temp
