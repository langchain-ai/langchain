"""Chain that calls Google Places API.

"""

import os
import sys
from typing import Any, Dict, Optional, Tuple

import aiohttp
from pydantic import BaseModel, Extra, Field, root_validator

from langchain.utils import get_from_dict_or_env

class GooglePlacesAPIWrapper(BaseModel):
    """Wrapper around Google Places API.

    To use, you should have the ``googlemaps`` python package installed, **an API key for the google maps platform**, and the enviroment variable ''GPLACES_API_KEY'' set with your API key , or pass 'gplaces_api_key' as a named parameter to the constructor. 

    By default, this will return contact the top-l results of an input search.

    Example:
        .. code-block:: python


            from langchain import GooglePlacesAPIWrapper
            gplaceapi = GooglePlacesAPIWrapper()
    """

    gplaces_api_key: Optional[str] = None
    google_map_client: Any #: :meta private:
    top_k_results: Optional[int] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True


    @root_validator()
    def validate_environment(cls, values:Dict) -> Dict:
        """Validate that api key is in your environment variable as ``GPLACES_API_KEY``."""
        gplaces_api_key = get_from_dict_or_env(
            values, "gplaces_api_key", "GPLACES_API_KEY"
        )
        values["gplaces_api_key"] = gplaces_api_key
        try:
            import googlemaps

            values['google_map_client'] = googlemaps.Client(gplaces_api_key)
        except ImportError:
            raise ValueError(
                "Could not import googlemaps python packge. "
                "Please install it with `pip install googlemaps`."
            )
        return values

    def run(self, query:str) -> str:
        """Run Places search and get k number of places that exists that match the description"""
        search_results = self.google_map_client.places(query)['results']
        search_results_len = len(search_results)

        places = []

        if search_results_len == 0:
            return "Google Places did not find any places that match the description"

        if self.top_k_results is None:
            for result in search_results:
                details = self.fetch_place_details(result['place_id'])

                if details is not None:
                    places.append(details)
        else:
            for i in range(min(search_results_len, self.top_k_results)):
                result = search_results[i]
                details = self.fetch_place_details(result['place_id'])

                if details is not None:
                    places.append(details)

        return "".join([f'{i+1}. {item}' for i, item in enumerate(places)])

    def fetch_place_details(self, place_id: str) -> Optional[str]:
        try:
            place_details = self.google_map_client.place(place_id)
            formatted_details = self.format_place_details(place_details)
            return formatted_details
        except Exception as e:
            print(f'An Error occured while fetching place details: {e}')
            return None

    def format_place_details(self, place_details: str) -> Optional[str]:
        try:
            name = place_details.get('result', {}).get('name', 'Unkown')
            address = place_details.get('result', {}).get('formatted_address', 'Unknown')
            phone_number = place_details.get('results', {}).get('formatted_phone_number', 'Unknown')
            website = place_details.get('results', {}).get('website', 'Unknown')


            formatted_details = f"{name}\nAddress: {address}\nPhone: {phone_number}\nWebsite: {website}\n\n"
            return formatted_details
        except Exception as e:
            print(f'An error occurred while formatting place details: {e}')
            return None





