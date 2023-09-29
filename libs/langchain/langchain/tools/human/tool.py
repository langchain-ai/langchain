"""Tool for asking human input."""

from typing import Callable, Optional

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.pydantic_v1 import Field
from langchain.tools.base import BaseTool
from pydantic import Field


def _print_func(text: str) -> None:
    print("\n")
    print(text)


class HumanInputRun(BaseTool):
    """Tool that asks user for input."""

    name: str = "human"
    description: str = (
        "You can ask a human for guidance when you think you "
        "got stuck or you are not sure what to do next. "
        "The input should be a question for the human, be sure to include sufficient context about what you've done, what tools you need access to if any, or anything else they might need to know to help you. Use best practices for asking good questions."
    )
    prompt_func: Callable[[str], None] = Field(default_factory=lambda: _print_func)
    input_func: Callable = Field(default_factory=lambda: input)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Human input tool."""
        self.prompt_func(query)
        return self.input_func()
