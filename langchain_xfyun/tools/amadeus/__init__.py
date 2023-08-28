"""Amadeus tools."""

from langchain_xfyun.tools.amadeus.closest_airport import AmadeusClosestAirport
from langchain_xfyun.tools.amadeus.flight_search import AmadeusFlightSearch

__all__ = [
    "AmadeusClosestAirport",
    "AmadeusFlightSearch",
]
