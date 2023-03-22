"""Tool for asking human input."""

import sys
from typing import Callable
from langchain.tools.base import BaseTool


class HumanInputRun(BaseTool):
    """Tool that adds the capability to ask user for input."""

    name = "Human"
    description = (
        "A human. "
        "Useful for clarification, confirmation, or anything other tools cannot help. "
        "Input should be a question you need help with."
    )
    prompt_func: Callable[[str], None]
    input_func: Callable

    def _run(self, query: str) -> str:
        """Use the Human input tool."""
        self.prompt_func(query)
        return self.input_func()

    async def _arun(self, query: str) -> str:
        """Use the Human tool asynchronously."""
        raise NotImplementedError("Human tool does not support async")
