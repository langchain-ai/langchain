"""Tools for interacting with the user."""


from typing import Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    RunManager,
)
from langchain.tools.base import BaseTool


class StdInInquireTool(BaseTool):
    """Tool for asking the user for input."""

    name: str = "Inquire"
    description: str = (
        "useful if you do not have enough information to"
        " effectively use other tools. Input is best as a clarifying"
        " question (to disambiguate) or a request for more context."
    )

    def _run(self, prompt: str, run_manager: Optional[RunManager] = None) -> str:
        """Prompt the user for more input."""
        return input(f"\n{prompt}")

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        raise NotImplementedError(f"{self.__class__.__name__} does not support async")
