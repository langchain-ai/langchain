from __future__ import annotations

from typing import TYPE_CHECKING, List

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.pydantic_v1 import Field
from langchain.tools import BaseTool
from langchain.tools.amadeus.closest_airport import AmadeusClosestAirport
from langchain.tools.amadeus.flight_search import AmadeusFlightSearch
from langchain.tools.amadeus.utils import authenticate

if TYPE_CHECKING:
    from amadeus import Client


class AmadeusToolkit(BaseToolkit):
    """Toolkit for interacting with Office365."""

    client: Client = Field(default_factory=authenticate)

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            AmadeusClosestAirport(),
            AmadeusFlightSearch(),
        ]
