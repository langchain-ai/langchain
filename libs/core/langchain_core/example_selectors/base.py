"""Interface for selecting examples to include in prompts."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_core.runnables import run_in_executor
from langchain_core.runnables.utils import gather_with_concurrency


class BaseExampleSelector(ABC):
    """Interface for selecting examples to include in prompts."""

    @abstractmethod
    def add_example(self, example: Dict[str, str]) -> Any:
        """Add new example to store.

        Args:
            example: A dictionary with keys as input variables
                and values as their values."""

    async def aadd_example(self, example: Dict[str, str]) -> Any:
        """Async add new example to store.

        Args:
            example: A dictionary with keys as input variables
                and values as their values."""

        return await run_in_executor(None, self.add_example, example)

    def add_examples(self, examples: List[Dict[str, str]]) -> Any:
        """Add new examples to store.

        Args:
            examples: A list of dictionaries mapping input variable names to their
                values.
        """
        results = []
        for example in examples:
            res = self.add_example(example)
            if res:
                results.append(res)
        return results or None

    async def aadd_examples(
        self, examples: List[Dict[str, str]], *, max_concurrency: Optional[int] = None
    ) -> Any:
        """Async add new examples to store.

        Args:
            examples: A list of dictionaries mapping input variable names to their
                values.
        """

        if self.__class__.add_examples != BaseExampleSelector.add_examples:
            return await run_in_executor(None, self.add_examples, examples)
        else:
            coros = map(self.aadd_example, examples)
            results = await gather_with_concurrency(max_concurrency, *coros)
            return results or None

    @abstractmethod
    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on the inputs.

        Args:
            input_variables: A dictionary with keys as input variables
                and values as their values."""

    async def aselect_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Async select which examples to use based on the inputs.

        Args:
            input_variables: A dictionary with keys as input variables
                and values as their values."""

        return await run_in_executor(None, self.select_examples, input_variables)
