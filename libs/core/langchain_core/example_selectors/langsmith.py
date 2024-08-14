"""Select examples using a LangSmith few-shot index."""
from typing import Any, Dict, List

from langchain_core.example_selectors.base import BaseExampleSelector


class LangSmithExampleSelector(BaseExampleSelector):
    """Select examples using a LangSmith few-shot index."""

    def __init__(
        self,

    ) -> None:

    def add_example(self, example: Dict[str, str]) -> Any:
        """Add new example to store.

        Args:
            example: A dictionary with keys as input variables
                and values as their values."""
        ...

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on the inputs.

        Args:
            input_variables: A dictionary with keys as input variables
                and values as their values."""
        ...

