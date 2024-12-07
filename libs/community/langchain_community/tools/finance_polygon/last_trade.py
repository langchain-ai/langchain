from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.finance_polygon import FinancePolygonAPIWrapper


class FinancePolygonLastTradeSchema(BaseModel):
    """Input for FinancePolygonLastTrade."""

    ticker: str = Field(description="The ticker symbol to fetch last trade data for.")


class FinancePolygonLastTrade(BaseTool):  # type: ignore[override, override]
    """
    Tool that provides the last trade data for a given stock.
    NOTE: this tool requires a "Stock Developer" subscription to Polygon.io.
    """

    mode: str = "get_last_trade"
    name: str = "polygon_last_trade"
    description: str = (
        "A wrapper around the Polygon's Stock API."
        "Get the last trade data for a given stock."
    )

    args_schema: Type[FinancePolygonLastTradeSchema] = FinancePolygonLastTradeSchema
    api_wrapper: FinancePolygonAPIWrapper

    def _run(
        self,
        ticker: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Polygon API tool."""
        return self.api_wrapper.run(
            mode=self.mode,
            ticker=ticker,
            run_manager=run_manager,
        )
