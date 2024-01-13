from typing import Dict, List

from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.tools import BaseTool
from langchain_community.tools.polygon import PolygonLastQuote
from langchain_community.tools.polygon.prompt import POLYGON_LAST_QUOTE
from langchain_community.utilities.polygon import PolygonAPIWrapper


class PolygonToolkit(BaseToolkit):
    """Polygon Toolkit."""

    tools: List[BaseTool] = []

    @classmethod
    def from_polygon_api_wrapper(
        cls, polygon_api_wrapper: PolygonAPIWrapper
    ) -> "PolygonToolkit":
        operations: List[Dict] = [
            {
                "mode": "get_last_quote",
                "name": "Get the last quote for a ticker",
                "description": POLYGON_LAST_QUOTE,
            },
        ]
        tools = [
            PolygonLastQuote(
                name=action["name"],
                description=action["description"],
                mode=action["mode"],
                api_wrapper=polygon_api_wrapper,
            )
            for action in operations
        ]
        return cls(tools=tools)

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return self.tools
