from abc import abstractmethod
from typing import List, Tuple

from pydantic import BaseModel, Field

from langchain.schema import BaseOutputParser


class Step(BaseModel):
    value: str


class Plan(BaseModel):
    steps: List[Step]


class StepResponse(BaseModel):
    response: str


class BaseStepContainer(BaseModel):
    @abstractmethod
    def add_step(self, step: Step, step_response: StepResponse) -> None:
        """Add step and step response to the container."""

    @abstractmethod
    def get_final_response(self) -> str:
        """Return the final response based on steps taken."""


class ListStepContainer(BaseModel):
    steps: List[Tuple[Step, StepResponse]] = Field(default_factory=list)

    def add_step(self, step: Step, step_response: StepResponse) -> None:
        self.steps.append((step, step_response))

    def get_steps(self) -> List[Tuple[Step, StepResponse]]:
        return self.steps

    def get_final_response(self) -> str:
        return self.steps[-1][1].response


class PlanOutputParser(BaseOutputParser):
    @abstractmethod
    def parse(self, text: str) -> Plan:
        """Parse into a plan."""
