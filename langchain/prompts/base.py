"""BasePrompt schema definition."""
from abc import ABC, abstractmethod
from typing import Any, List

from langchain.formatting import formatter

DEFAULT_FORMATTER_MAPPING = {
    "f-string": formatter.format,
}


class BasePrompt(ABC):
    """Base prompt should expose the format method, returning a prompt."""

    input_variables: List[str]
    """A list of the names of the variables the prompt template expects."""

    @abstractmethod
    def format(self, **kwargs: Any) -> str:
        """Format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.

        Example:

        .. code-block:: python

            prompt.format(variable1="foo")
        """
