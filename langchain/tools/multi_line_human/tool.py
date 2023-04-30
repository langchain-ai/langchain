"""Tool for asking human input."""

from typing import Callable

from pydantic import Field

from langchain.tools.base import BaseTool


def _print_func(text: str) -> None:
    print("\n")
    print(text)

def input_func():
    print("Insert your text. Press Ctrl-D (or Ctrl-Z on Windows) to end.")
    contents = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        contents.append(line)
    return "\n".join(contents)

class MultiLineHumanInputRun(BaseTool):
    """Tool that adds the capability to ask user for multi line input."""

    name = "MultiLineHuman"
    description = (
        "You can ask a human for guidance when you think you"
        " got stuck or you are not sure what to do next."
        " The input should be a question for the human."
        " This tool version is suitable when you need answers that span over"
        " several lines."
    )
    prompt_func: Callable[[str], None] = Field(default_factory=lambda: _print_func)

    input_func: Callable = Field(default_factory=lambda: input_func)

    def _run(self, query: str) -> str:
        """Use the Multi Line Human input tool."""
        self.prompt_func(query)
        return self.input_func()

    async def _arun(self, query: str) -> str:
        """Use the Multi Line Human tool asynchronously."""
        raise NotImplementedError("Human tool does not support async")
