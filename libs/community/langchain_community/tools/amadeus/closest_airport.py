from typing import Any, Dict, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.language_models import BaseLanguageModel
from pydantic import BaseModel, Field, model_validator

from langchain_community.chat_models import ChatOpenAI
from langchain_community.tools.amadeus.base import AmadeusBaseTool


class ClosestAirportSchema(BaseModel):
    """Schema for the AmadeusClosestAirport tool."""

    location: str = Field(
        description=(
            " The location for which you would like to find the nearest airport "
            " along with optional details such as country, state, region, or "
            " province, allowing for easy processing and identification of "
            " the closest airport. Examples of the format are the following:\n"
            " Cali, Colombia\n "
            " Lincoln, Nebraska, United States\n"
            " New York, United States\n"
            " Sydney, New South Wales, Australia\n"
            " Rome, Lazio, Italy\n"
            " Toronto, Ontario, Canada\n"
        )
    )


class AmadeusClosestAirport(AmadeusBaseTool):  # type: ignore[override, override, override]
    """Tool for finding the closest airport to a particular location."""

    name: str = "closest_airport"
    description: str = (
        "Use this tool to find the closest airport to a particular location."
    )
    args_schema: Type[ClosestAirportSchema] = ClosestAirportSchema

    llm: Optional[BaseLanguageModel] = Field(default=None)
    """Tool's llm used for calculating the closest airport. Defaults to `ChatOpenAI`."""

    @model_validator(mode="before")
    @classmethod
    def set_llm(cls, values: Dict[str, Any]) -> Any:
        if not values.get("llm"):
            # For backward-compatibility
            values["llm"] = ChatOpenAI(temperature=0)
        return values

    def _run(
        self,
        location: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        content = (
            f" What is the nearest airport to {location}? Please respond with the "
            " airport's International Air Transport Association (IATA) Location "
            ' Identifier in the following JSON format. JSON: "iataCode": "IATA '
            ' Location Identifier" '
        )

        return self.llm.invoke(content)  # type: ignore[union-attr]
