from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from langchain_community.utilities.polygon import PolygonAPIWrapper


class Inputs(BaseModel):
    """Inputs for Polygon's Financials API"""

    query: str


class PolygonFinancials(BaseTool):  # type: ignore[override, override]
    """Tool that gets the financials of a ticker from Polygon"""

    mode: str = "get_financials"
    name: str = "polygon_financials"
    description: str = (
        "A wrapper around Polygon's Stock Financials API. "
        "This tool is useful for fetching fundamental financials from "
        "balance sheets, income statements, and cash flow statements "
        "for a stock ticker. The input should be the ticker that you want "
        "to get the latest fundamental financial data for."
    )
    args_schema: Type[BaseModel] = Inputs

    api_wrapper: PolygonAPIWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Polygon API tool."""
        return self.api_wrapper.run(self.mode, ticker=query)
