"""Agent state data."""

from abc import ABC, abstractmethod
from typing import Dict, List, Sequence, Tuple

from pydantic import BaseModel

from langchain.agents.step import StepOutput
from langchain.schema import AgentAction


class BaseAgentMemory(BaseModel, ABC):
    """Agent working memory, modeled off of Chain memory.

    Should contain everything needed for the agent to make its next decision. After
    all, how can what you don't remember possibly inform your decisions?
    """

    inputs: Dict[str, str]

    @abstractmethod
    def add_step(self, output: StepOutput) -> None:
        """Add step output to memory."""

    @abstractmethod
    def steps(self) -> Sequence[StepOutput]:
        """Return all steps so far.

        Uses Sequence instead of List so that this could return subclasses of
        StepOutput. This is because List is invariant but Sequence is covariant:

        https://mypy.readthedocs.io/en/stable/common_issues.html#variance
        """

    @abstractmethod
    def as_intermediate_steps(self) -> List[Tuple[AgentAction, str]]:
        """Return intermediate steps in unstructured form.

        For backwards compatibility.
        """


class AgentMemory(BaseAgentMemory):
    """Default Agent memory that assumes StepOutput type for all steps."""

    _steps: List[StepOutput] = []

    def add_step(self, output: StepOutput) -> None:
        """Add step output to memory."""
        self._steps.append(output)

    def steps(self) -> Sequence[StepOutput]:
        """Return all steps so far."""
        return self._steps

    def as_intermediate_steps(self) -> List[Tuple[AgentAction, str]]:
        """Return intermediate steps in unstructured form.

        For backwards compatibility.
        """
        return [output.as_intermediate_step() for output in self._steps]
