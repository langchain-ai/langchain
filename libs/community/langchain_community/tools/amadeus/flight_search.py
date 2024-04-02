import logging
from datetime import datetime as dt
from typing import Dict, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_community.tools.amadeus.base import AmadeusBaseTool

logger = logging.getLogger(__name__)


class FlightSearchSchema(BaseModel):
    """Schema for the AmadeusFlightSearch tool."""

    originLocationCode: str = Field(
        description=(
            " The three letter International Air Transport "
            " Association (IATA) Location Identifier for the "
            " search's origin airport. "
        )
    )
    destinationLocationCode: str = Field(
        description=(
            " The three letter International Air Transport "
            " Association (IATA) Location Identifier for the "
            " search's destination airport. "
        )
    )
    departureDateTimeEarliest: str = Field(
        description=(
            " The earliest departure datetime from the origin airport "
            " for the flight search in the following format: "
            ' "YYYY-MM-DDTHH:MM:SS", where "T" separates the date and time '
            ' components. For example: "2023-06-09T10:30:00" represents '
            " June 9th, 2023, at 10:30 AM. "
        )
    )
    departureDateTimeLatest: str = Field(
        description=(
            " The latest departure datetime from the origin airport "
            " for the flight search in the following format: "
            ' "YYYY-MM-DDTHH:MM:SS", where "T" separates the date and time '
            ' components. For example: "2023-06-09T10:30:00" represents '
            " June 9th, 2023, at 10:30 AM. "
        )
    )
    page_number: int = Field(
        default=1,
        description="The specific page number of flight results to retrieve",
    )


class AmadeusFlightSearch(AmadeusBaseTool):
    """Tool for searching for a single flight between two airports."""

    name: str = "single_flight_search"
    description: str = (
        " Use this tool to search for a single flight between the origin and "
        " destination airports at a departure between an earliest and "
        " latest datetime. "
    )
    args_schema: Type[FlightSearchSchema] = FlightSearchSchema

    def _run(
        self,
        originLocationCode: str,
        destinationLocationCode: str,
        departureDateTimeEarliest: str,
        departureDateTimeLatest: str,
        page_number: int = 1,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> list:
        try:
            from amadeus import ResponseError
        except ImportError as e:
            raise ImportError(
                "Unable to import amadeus, please install with `pip install amadeus`."
            ) from e

        RESULTS_PER_PAGE = 10

        # Authenticate and retrieve a client
        client = self.client

        # Check that earliest and latest dates are in the same day
        earliestDeparture = dt.strptime(departureDateTimeEarliest, "%Y-%m-%dT%H:%M:%S")
        latestDeparture = dt.strptime(departureDateTimeLatest, "%Y-%m-%dT%H:%M:%S")

        if earliestDeparture.date() != latestDeparture.date():
            logger.error(
                " Error: Earliest and latest departure dates need to be the "
                " same date. If you're trying to search for round-trip "
                " flights, call this function for the outbound flight first, "
                " and then call again for the return flight. "
            )
            return [None]

        # Collect all results from the Amadeus Flight Offers Search API
        response = None
        try:
            response = client.shopping.flight_offers_search.get(
                originLocationCode=originLocationCode,
                destinationLocationCode=destinationLocationCode,
                departureDate=latestDeparture.strftime("%Y-%m-%d"),
                adults=1,
            )
        except ResponseError as error:
            print(error)  # noqa: T201

        # Generate output dictionary
        output = []
        if response is not None:
            for offer in response.data:
                itinerary: Dict = {}
                itinerary["price"] = {}
                itinerary["price"]["total"] = offer["price"]["total"]
                currency = offer["price"]["currency"]
                currency = response.result["dictionaries"]["currencies"][currency]
                itinerary["price"]["currency"] = {}
                itinerary["price"]["currency"] = currency

                segments = []
                for segment in offer["itineraries"][0]["segments"]:
                    flight = {}
                    flight["departure"] = segment["departure"]
                    flight["arrival"] = segment["arrival"]
                    flight["flightNumber"] = segment["number"]
                    carrier = segment["carrierCode"]
                    carrier = response.result["dictionaries"]["carriers"][carrier]
                    flight["carrier"] = carrier

                    segments.append(flight)

                itinerary["segments"] = []
                itinerary["segments"] = segments

                output.append(itinerary)

        # Filter out flights after latest departure time
        for index, offer in enumerate(output):
            offerDeparture = dt.strptime(
                offer["segments"][0]["departure"]["at"], "%Y-%m-%dT%H:%M:%S"
            )

            if offerDeparture > latestDeparture:
                output.pop(index)

        # Return the paginated results
        startIndex = (page_number - 1) * RESULTS_PER_PAGE
        endIndex = startIndex + RESULTS_PER_PAGE

        return output[startIndex:endIndex]
