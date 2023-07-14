"""Tool for the Wolfram Alpha API."""

from typing import Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.base import BaseTool
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper


class WolframAlphaQueryRun(BaseTool):
    """Tool that adds the capability to query using the Wolfram Alpha SDK."""

    name = "wolfram_alpha"
    description = (
        "A wrapper around Wolfram Alpha. "
        "Useful for when you need to answer questions about Math, "
        "Science, Technology, Culture, Society and Everyday Life. "
        "Input should be a search query."
    )
    api_wrapper: WolframAlphaAPIWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the WolframAlpha tool."""
        return self.api_wrapper.run(query)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the WolframAlpha tool asynchronously."""
        raise NotImplementedError("WolframAlphaQueryRun does not support async")
