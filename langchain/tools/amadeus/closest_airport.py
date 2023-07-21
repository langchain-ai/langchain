from typing import Optional, Type

from pydantic import BaseModel, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.tools.amadeus.base import AmadeusBaseTool


class ClosestAirportSchema(BaseModel):
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


class AmadeusClosestAirport(AmadeusBaseTool):
    name: str = "closest_airport"
    description: str = (
        "Use this tool to find the closest airport to a particular location."
    )
    args_schema: Type[ClosestAirportSchema] = ClosestAirportSchema

    def _run(
        self,
        location: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        template = (
            " What is the nearest airport to {location}? Please respond with the "
            " airport's International Air Transport Association (IATA) Location "
            ' Identifier in the following JSON format. JSON: "iataCode": "IATA '
            ' Location Identifier" '
        )

        llm = ChatOpenAI(temperature=0)

        llm_chain = LLMChain.from_string(llm=llm, template=template)

        output = llm_chain.run(location=location)

        return output

    async def _arun(
        self,
        location: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        raise NotImplementedError(f"The tool {self.name} does not support async yet.")
