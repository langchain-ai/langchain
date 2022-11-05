"""Base interface that all chains should implement."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pydantic import BaseModel


class Chain(BaseModel, ABC):
    """Base interface that all chains should implement."""

    verbose: bool = False
    """Whether to print out the code that was executed."""

    @property
    @abstractmethod
    def input_keys(self) -> List[str]:
        """Input keys this chain expects."""

    @property
    @abstractmethod
    def output_keys(self) -> List[str]:
        """Output keys this chain expects."""

    def _validate_inputs(self, inputs: Dict[str, str]) -> None:
        """Check that all inputs are present."""
        missing_keys = set(self.input_keys).difference(inputs)
        if missing_keys:
            raise ValueError(f"Missing some input keys: {missing_keys}")

    def _validate_outputs(self, outputs: Dict[str, str]) -> None:
        if set(outputs) != set(self.output_keys):
            raise ValueError(
                f"Did not get output keys that were expected. "
                f"Got: {set(outputs)}. Expected: {set(self.output_keys)}."
            )

    @abstractmethod
    def _run(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Run the logic of this chain and return the output."""

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Run the logic of this chain and add to output."""
        self._validate_inputs(inputs)
        if self.verbose:
            print("\n\n\033[1m> Entering new chain...\033[0m")
        outputs = self._run(inputs)
        if self.verbose:
            print("\n\033[1m> Finished chain.\033[0m")
        self._validate_outputs(outputs)
        return {**inputs, **outputs}
