from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.finance_polygon import FinancePolygonAPIWrapper


class FinancePolygonMACDSchema(BaseModel):
    """Input for Polygon MACD (Moving Average Convergence/Divergence)."""

    ticker: str = Field(
        description="The ticker symbol of the stock to get the MACD data for."
    )
    timespan: str = Field(
        default="day",
        description="The size of the aggregate time window. Default is 'day'.",
    )
    adjusted: bool = Field(
        default=True, description="Whether to adjust for splits. Default is true."
    )
    short_window: int = Field(
        default=12,
        description="The short window size for MACD calculation. Default is 12.",
    )
    long_window: int = Field(
        default=26,
        description="The long window size for MACD calculation. Default is 26.",
    )
    signal_window: int = Field(
        default=9, description="The signal line window size for MACD. Default is 9."
    )
    series_type: str = Field(
        default="close",
        description="Price type to calculate MACD, e.g., 'close'. Default is 'close'.",
    )
    order: str = Field(
        default="desc", description="Order of results by timestamp. Default is 'desc'."
    )
    limit: int = Field(
        default=10,
        description="Number of results to return. Default is 10, max is 5000.",
    )


class PolygonMACD(BaseTool):  # type: ignore[override, override]
    """
    Tool for fetching Moving Average Convergence/Divergence (MACD) data for a ticker.
    """

    mode: str = "get_macd"
    name: str = "polygon_macd"
    description: str = (
        "A wrapper around Polygon's MACD API. "
        "This tool provides the MACD data for a given stock ticker."
    )

    args_schema: Type[FinancePolygonMACDSchema] = FinancePolygonMACDSchema
    api_wrapper: FinancePolygonAPIWrapper

    def _run(
        self,
        ticker: str,
        timespan: str,
        adjusted: bool,
        short_window: int,
        long_window: int,
        signal_window: int,
        series_type: str,
        order: str,
        limit: int,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Polygon API tool."""
        return self.api_wrapper.run(
            mode=self.mode,
            ticker=ticker,
            timespan=timespan,
            adjusted=adjusted,
            short_window=short_window,
            long_window=long_window,
            signal_window=signal_window,
            series_type=series_type,
            order=order,
            limit=limit,
            run_manager=run_manager,
        )
