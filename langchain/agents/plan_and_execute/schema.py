from typing import List

from pydantic import BaseModel

from langchain.schema import BaseOutputParser


class Plan(BaseModel):
    steps: List[str]


class Step(BaseModel):
    response: str


class PlanOutputParser(BaseOutputParser):
    def parse(self, text: str) -> Plan:
        """Parse into a plan."""
