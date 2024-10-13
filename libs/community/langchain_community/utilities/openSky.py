from datetime import datetime
from typing import Any, Dict, List

import requests
from pydantic import BaseModel


class OpenSkyAPIWrapper(BaseModel):
    """Wrapper for the OpenSky API."""

    def fetch_all_state_vectors(self) -> List[Dict[str, Any]]:
        """Fetch all state vectors from OpenSky API and parse the data."""
        response = requests.get("https://opensky-network.org/api/states/all")
        if response.status_code == 200:
            json_data = response.json()
            states = json_data.get("states", [])
            return self.parse_state_vectors(json_data["time"], states)
        else:
            return {"error": response.text}

    def fetch_flights_by_time_interval(
        self, begin: int, end: int
    ) -> List[Dict[str, Any]]:
        """Fetch flight data for a specified time interval."""
        url = f"https://opensky-network.org/api/flights/all?begin={begin}&end={end}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if not data:
                return {"error": "No flights found for the specified time interval."}
            return self.parse_flight_data(data)
        else:
            return {"error": response.text}

    def fetch_flights_by_aircraft(
        self, icao24: str, begin: int, end: int
    ) -> List[Dict[str, Any]]:
        """Fetch flights for a specific aircraft within a time interval."""
        url = f"https://opensky-network.org/api/flights/aircraft?icao24={icao24}&begin={begin}&end={end}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if not data:
                return {"error": "No flights found for the specified aircraft."}
            return self.parse_flight_data(data)
        else:
            return {"error": response.text}

    def fetch_arrivals_by_airport(
        self, airport: str, begin: int, end: int
    ) -> List[Dict[str, Any]]:
        """Fetch arrivals by airport within a time interval."""
        url = f"https://opensky-network.org/api/flights/arrival?airport={airport}&begin={begin}&end={end}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if not data:
                return {"error": "No arrivals found for the specified airport."}
            return self.parse_flight_data(data)
        else:
            return {"error": response.text}

    def fetch_departures_by_airport(
        self, airport: str, begin: int, end: int
    ) -> List[Dict[str, Any]]:
        """Fetch departures by airport within a time interval."""
        url = f"https://opensky-network.org/api/flights/departure?airport={airport}&begin={begin}&end={end}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if not data:
                return {"error": "No departures found for the specified airport."}
            return self.parse_flight_data(data)
        else:
            return {"error": response.text}

    def parse_flight_data(
        self, flight_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Parse flight data from the API response."""
        parsed_flights = []
        for flight in flight_data:
            parsed_flight = {
                "ICAO 24 Address": flight.get("icao24"),
                "First Seen": datetime.fromtimestamp(
                    flight.get("firstSeen")
                ).isoformat(),
                "Estimated Departure Airport": flight.get("estDepartureAirport"),
                "Last Seen": datetime.fromtimestamp(flight.get("lastSeen")).isoformat(),
                "Estimated Arrival Airport": flight.get("estArrivalAirport"),
                "Callsign": flight.get("callsign"),
                "Departure Candidates": flight.get("departureAirportCandidatesCount"),
                "Arrival Candidates": flight.get("arrivalAirportCandidatesCount"),
            }
            parsed_flights.append(parsed_flight)
        return parsed_flights

    def parse_state_vectors(
        self, timestamp: int, states: List[List[Any]]
    ) -> List[Dict[str, Any]]:
        """Parse state vector data from the API response."""
        parsed_states = []
        # Define a mapping for position sources
        position_source_mapping = {0: "ADS-B", 1: "ASTERIX", 2: "MLAT", 3: "FLARM"}
        for state in states:
            parsed_state = {
                "ICAO 24 Address": state[0],  # icao24
                "Callsign": state[1].strip() if state[1] else None,  # callsign
                "Country": state[2],  # origin_country
                "Last Position Update": (
                    datetime.fromtimestamp(state[3]).isoformat() if state[3] else None
                ),
                "Last Contact": (
                    datetime.fromtimestamp(state[4]).isoformat() if state[4] else None
                ),
                "Longitude": state[5],  # longitude
                "Latitude": state[6],  # latitude
                "Barometric Altitude": state[7],  # baro_altitude
                "On Ground": state[8],  # on_ground
                "Velocity": state[9],  # velocity
                "True Track": state[10],  # true_track
                "Vertical Rate": state[11],  # vertical_rate
                "Sensors": state[12],  # sensors
                "Geometric Altitude": state[13],  # geo_altitude
                "Squawk": state[14],  # squawk
                "Special Purpose Indicator": state[15],  # spi
                "Position Source": position_source_mapping.get(state[16], "Unknown"),
            }
            # Check if there is a category field; if not, set it to None
            if len(state) > 17:
                parsed_state["Category"] = state[17]  # category
            else:
                parsed_state["Category"] = None  # Default to None if missing

            parsed_states.append(parsed_state)
        return parsed_states
