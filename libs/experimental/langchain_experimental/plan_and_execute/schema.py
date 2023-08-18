from abc import abstractmethod
from typing import List, Tuple

from langchain.schema import BaseOutputParser

from langchain_experimental.pydantic_v1 import BaseModel, Field


class Step(BaseModel):
    """Step."""

    value: str
    """The value."""


class Plan(BaseModel):
    """Plan."""

    steps: List[Step]
    """The steps."""


class StepResponse(BaseModel):
    """Step response."""

    response: str
    """The response."""


class BaseStepContainer(BaseModel):
    """Base step container."""

    @abstractmethod
    def add_step(self, step: Step, step_response: StepResponse) -> None:
        """Add step and step response to the container."""

    @abstractmethod
    def get_final_response(self) -> str:
        """Return the final response based on steps taken."""


class ListStepContainer(BaseStepContainer):
    """List step container."""

    steps: List[Tuple[Step, StepResponse]] = Field(default_factory=list)
    """The steps."""

    def add_step(self, step: Step, step_response: StepResponse) -> None:
        self.steps.append((step, step_response))

    def get_steps(self) -> List[Tuple[Step, StepResponse]]:
        return self.steps

    def get_final_response(self) -> str:
        return self.steps[-1][1].response


class PlanOutputParser(BaseOutputParser):
    """Plan output parser."""

    @abstractmethod
    def parse(self, text: str) -> Plan:
        """Parse into a plan."""
