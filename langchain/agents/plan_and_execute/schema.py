from abc import abstractmethod
from typing import List

from pydantic import BaseModel

from langchain.schema import BaseOutputParser


class Step(BaseModel):
    value: str


class Plan(BaseModel):
    steps: List[Step]


class StepResponse(BaseModel):
    response: str


class PlanOutputParser(BaseOutputParser):
    @abstractmethod
    def parse(self, text: str) -> Plan:
        """Parse into a plan."""
