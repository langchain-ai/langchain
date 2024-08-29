from typing import List

from langchain_core.tools import BaseTool
from langchain_core.tools.base import BaseToolkit

from langchain_community.tools.polygon import (
    PolygonAggregates,
    PolygonFinancials,
    PolygonLastQuote,
    PolygonTickerNews,
)
from langchain_community.utilities.polygon import PolygonAPIWrapper


class PolygonToolkit(BaseToolkit):
    """Polygon Toolkit.

    Parameters:
        tools: List[BaseTool]. The tools in the toolkit.
    """

    tools: List[BaseTool] = []

    @classmethod
    def from_polygon_api_wrapper(
        cls, polygon_api_wrapper: PolygonAPIWrapper
    ) -> "PolygonToolkit":
        """Create a Polygon Toolkit from a Polygon API Wrapper.

        Args:
            polygon_api_wrapper: PolygonAPIWrapper. The Polygon API Wrapper.

        Returns:
            PolygonToolkit. The Polygon Toolkit.
        """
        tools = [
            PolygonAggregates(
                api_wrapper=polygon_api_wrapper,
            ),
            PolygonLastQuote(
                api_wrapper=polygon_api_wrapper,
            ),
            PolygonTickerNews(
                api_wrapper=polygon_api_wrapper,
            ),
            PolygonFinancials(
                api_wrapper=polygon_api_wrapper,
            ),
        ]
        return cls(tools=tools)

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return self.tools
