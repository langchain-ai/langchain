"""Chain pipeline where the given chain is run repeatedly until a terminating condition is met."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain


class BaseWhileChain(Chain, BaseModel, ABC):
    """Inherit to run a chain on while loop while updating its input every iteration, stopping when the output condition is met"""

    chain: Chain
    input_variables: List[str]
    output_key: str = "output"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return self.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Return output key.

        :meta private:
        """
        return [self.output_key]

    @abstractmethod
    def get_initial_state(self, inputs: Dict[str, str]) -> Any:
        """Given the input, computes the initial state

        Args:
            inputs: the dictionary of inputs to the the chain

        Returns:
            An object that represents the initial state at which the loop starts
        """

    @abstractmethod
    def get_updated_state(self, current_state: Any, inputs: Dict[str, str]) -> Any:
        """Given the output after a iteration, computes the updated state

        Args:
            current_state: the state object
            inputs: the dictionary of inputs to the the chain

        Returns:
            Updated state object
        """

    @abstractmethod
    def stopping_criterion(self, outputs: Dict[str, str]) -> bool:
        """Evaluates the current iteration's output to see if the loop should be terminated"""

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        state = self.get_initial_state(inputs)
        while True:
            outputs = self.chain(state)
            if self.stopping_criterion(outputs):
                break
            state = self.get_updated_state(state, inputs)
        return {self.output_key: outputs}
