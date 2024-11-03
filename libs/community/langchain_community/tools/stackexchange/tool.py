"""Tool for the Wikipedia API."""

from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

from langchain_community.utilities.stackexchange import StackExchangeAPIWrapper


class StackExchangeTool(BaseTool):  # type: ignore[override]
    """Tool that uses StackExchange"""

    name: str = "stack_exchange"
    description: str = (
        "A wrapper around StackExchange. "
        "Useful for when you need to answer specific programming questions"
        "code excerpts, code examples and solutions"
        "Input should be a fully formed question."
    )
    api_wrapper: StackExchangeAPIWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Stack Exchange tool."""
        return self.api_wrapper.run(query)
