from typing import List

from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.tools import BaseTool
from langchain_community.tools.polygon import PolygonLastQuote, PolygonTickerNews
from langchain_community.utilities.polygon import PolygonAPIWrapper


class PolygonToolkit(BaseToolkit):
    """Polygon Toolkit."""

    tools: List[BaseTool] = []

    @classmethod
    def from_polygon_api_wrapper(
        cls, polygon_api_wrapper: PolygonAPIWrapper
    ) -> "PolygonToolkit":
        tools = [
            PolygonLastQuote(
                api_wrapper=polygon_api_wrapper,
            ),
            PolygonTickerNews(
                api_wrapper=polygon_api_wrapper,
            ),
        ]
        return cls(tools=tools)

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return self.tools
