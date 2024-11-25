from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.finance_polygon import FinancePolygonAPIWrapper


class FinancePolygonSMASchema(BaseModel):
    """Input for Polygon SMA (Simple Moving Average)."""

    ticker: str = Field(
        description="The ticker symbol of the stock to get the SMA data for."
    )
    timespan: str = Field(
        default="day",
        description="The size of the aggregate time window. Default is 'day'.",
    )
    adjusted: bool = Field(
        default=True, description="Whether to adjust for splits. Default is true."
    )
    window: int = Field(
        default=50, description="Window size for SMA calculation. Default is 50."
    )
    series_type: str = Field(
        default="close",
        description="Price type to calculate SMA, e.g., 'close'. Default is 'close'.",
    )
    order: str = Field(
        default="desc", description="Order of results by timestamp. Default is 'desc'."
    )
    limit: int = Field(
        default=10,
        description="Number of results to return. Default is 10, max is 5000.",
    )


class PolygonSMA(BaseTool):  # type: ignore[override, override]
    """
    Tool for fetching Simple Moving Average (SMA) data for a ticker.
    """

    mode: str = "get_sma"
    name: str = "polygon_sma"
    description: str = (
        "A wrapper around Polygon's SMA API. "
        "This tool provides the simple moving average (SMA) for a given stock ticker."
    )

    args_schema: Type[FinancePolygonSMASchema] = FinancePolygonSMASchema
    api_wrapper: FinancePolygonAPIWrapper

    def _run(
        self,
        ticker: str,
        timespan: str,
        adjusted: bool,
        window: int,
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
            window=window,
            series_type=series_type,
            order=order,
            limit=limit,
            run_manager=run_manager,
        )
