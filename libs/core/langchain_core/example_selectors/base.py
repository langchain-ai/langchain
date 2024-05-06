"""Interface for selecting examples to include in prompts."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from langchain_core.runnables import run_in_executor


class BaseExampleSelector(ABC):
    """Interface for selecting examples to include in prompts."""

    @abstractmethod
    def add_example(self, example: Dict[str, str]) -> Any:
        """Add new example to store."""

    async def aadd_example(self, example: Dict[str, str]) -> Any:
        """Add new example to store."""
        return await run_in_executor(None, self.add_example, example)

    @abstractmethod
    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on the inputs."""

    async def aselect_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on the inputs."""
        return await run_in_executor(None, self.select_examples, input_variables)
