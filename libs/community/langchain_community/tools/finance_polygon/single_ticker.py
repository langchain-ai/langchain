from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.finance_polygon import FinancePolygonAPIWrapper


class FinancePolygonSingleTickerSchema(BaseModel):
    """Input for Polygon Single Ticker."""

    ticker: str = Field(
        description="The ticker symbol of the stock to get market data for."
    )


class PolygonSingleTicker(BaseTool):  # type: ignore[override, override]
    """
    Tool for fetching market data for a single traded stock ticker.
    """

    mode: str = "get_single_ticker"
    name: str = "polygon_single_ticker"
    description: str = (
        "A wrapper around Polygon's Single Ticker API. "
        "This tool provides the most up-to-date market data"
        "for a specific traded stock symbol."
    )

    args_schema: Type[FinancePolygonSingleTickerSchema] = (
        FinancePolygonSingleTickerSchema
    )
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
