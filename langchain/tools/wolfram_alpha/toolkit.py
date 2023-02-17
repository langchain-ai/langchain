"""Toolkit for the Wolfram Alpha API."""

from typing import List

from langchain.tools.base import BaseTool, BaseToolkit
from langchain.tools.wolfram_alpha.tool import WolframAlphaQueryRun
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper


class WolframAlphaToolkit(BaseToolkit):
    """Tool that adds the capability to interact with Wolfram Alpha."""

    wolfram_alpha_appid: str

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        wrapper = WolframAlphaAPIWrapper(wolfram_alpha_appid=self.wolfram_alpha_appid)
        return [
            WolframAlphaQueryRun(
                api_wrapper=wrapper,
            )
        ]
