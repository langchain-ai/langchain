"""Tool for the Wikipedia API."""

from typing import Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.base import BaseTool
from langchain.utilities.wikipedia import WikipediaAPIWrapper


class WikipediaQueryRun(BaseTool):
    """Tool that searches the Wikipedia API."""

    name = "Wikipedia"
    description = (
        "A wrapper around Wikipedia. "
        "Useful for when you need to answer general questions about "
        "people, places, companies, facts, historical events, or other subjects. "
        "Input should be a search query."
    )
    api_wrapper: WikipediaAPIWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Wikipedia tool."""
        return self.api_wrapper.run(query)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Wikipedia tool asynchronously."""
        raise NotImplementedError("WikipediaQueryRun does not support async")
