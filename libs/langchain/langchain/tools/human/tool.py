"""Tool for asking human input."""

from typing import Callable, Optional

from langchain_core.pydantic_v1 import Field

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.base import BaseTool


def _print_func(text: str) -> None:
    print("\n")
    print(text)


class HumanInputRun(BaseTool):
    """Tool that asks user for input."""

    name: str = "human"
    description: str = (
        "Ты можешь попросить человека о помощи, когда ты думаешь, что "
        "застрял или не уверен, что делать дальше. "
        "Ввод должен быть вопросом для человека."
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
