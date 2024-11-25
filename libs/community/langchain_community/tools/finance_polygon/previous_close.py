from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.finance_polygon import FinancePolygonAPIWrapper


class PolygonPreviousCloseSchema(BaseModel):
    """Inputs for PolygonPreviousClose."""

    ticker: str = Field(description="The ticker symbol to fetch previous close for.")


class PolygonPreviousClose(BaseTool):  # type: ignore[override, override]
    """
    Tool that gets the previous day's open, high, low,
    and close (OHLC) for the specified stock ticker.

    """

    mode: str = "get_previous_close"
    name: str = "polygon_previous_close"
    description: str = (
        "A wrapper around Polygon's Previous Close API. "
        "This tool is useful for fetching the previous day's open, high"
        "low, and close (OHLC). Input should be the ticker that you want to"
        "get the previous close for."
    )
    args_schema: Type[PolygonPreviousCloseSchema] = PolygonPreviousCloseSchema
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
        )
