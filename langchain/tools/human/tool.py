"""Tool for asking human input."""
import asyncio
from typing import Callable, Optional

from pydantic import Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.base import BaseTool


async def _async_input_func(input_func: Callable) -> str:
    if asyncio.iscoroutinefunction(input_func):
        # If the input_func is async, await it directly
        return await input_func()
    else:
        # If the input_func is synchronous, run it in an executor
        return await asyncio.get_event_loop().run_in_executor(None, input_func)


def _print_func(text: str) -> None:
    print("\n")
    print(text)


class HumanInputRun(BaseTool):
    """Tool that adds the capability to ask user for input."""

    name = "human"
    description = (
        "You can ask a human for guidance when you think you "
        "got stuck or you are not sure what to do next. "
        "The input should be a question for the human."
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

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Human tool asynchronously."""
        if asyncio.iscoroutinefunction(self.prompt_func):
            # If the prompt_func is async, await it directly
            await self.prompt_func(query)
        else:
            # If the prompt_func is synchronous, run it in an executor
            await asyncio.get_event_loop().run_in_executor(
                None, self.prompt_func, query
            )

        # Use the asynchronous input function.
        return await _async_input_func(self.input_func)
