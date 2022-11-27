"""Handle chained inputs."""
from typing import Optional

from langchain.logger import Logger


class ChainedInput:
    """Class for working with input that is the result of chains."""

    def __init__(self, text: str, context: dict, logger: Optional[Logger] = None):
        """Initialize with verbose flag and initial text."""
        self._logger = logger
        if self._logger:
            self._logger.log(text, context)
        self._input = text
        self._context = context

    def add(self, text: str, color: Optional[str] = None) -> None:
        """Add text to input, print if in verbose mode."""
        if self._logger:
            self._logger.log(text, self._context, color=color)
        self._input += text

    @property
    def input(self) -> str:
        """Return the accumulated input."""
        return self._input
