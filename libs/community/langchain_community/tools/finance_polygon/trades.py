from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.finance_polygon import FinancePolygonAPIWrapper


class PolygonTradesSchema(BaseModel):
    """Inputs for PolygonTrades."""

    ticker: str = Field(
        description="The ticker symbol to fetch trades for.",
    )


class PolygonTrades(BaseTool):  # type: ignore[override, override]
    """
    Tool that gets trades for a ticker symbol in a given timestamp.
    """

    mode: str = "get_trades"
    name: str = "polygon_trades"
    description: str = (
        "A wrapper around Polygon's Trades API. "
        "This tool is useful for fetching the trades for a ticker."
        "Input should be the ticker symbol that you want to"
        "get the trades for."
    )
    args_schema: Type[PolygonTradesSchema] = PolygonTradesSchema
    api_wrapper: FinancePolygonAPIWrapper

    def _run(
        self,
        stocksTicker: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Polygon API tool."""
        return self.api_wrapper.run(
            mode=self.mode,
            stocksTicker=stocksTicker,
        )
