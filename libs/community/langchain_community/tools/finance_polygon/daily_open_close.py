from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.finance_polygon import FinancePolygonAPIWrapper


class PolygonDailyOpenCloseSchema(BaseModel):
    """Inputs for PolygonDailyOpenClose."""

    stocksTicker: str = Field(
        description="The ticker symbol to fetch daily/open close for.",
    )
    date: str = Field(
        description="The date of the open/close with format YYYY-MM-DD",
    )


class PolygonDailyOpenClose(BaseTool):  # type: ignore[override, override]
    """
    Tool that gets the daily open/close for a given
    ticker and date from Polygon
    """

    mode: str = "get_daily_open_close"
    name: str = "polygon_daily_open_close"
    description: str = (
        "A wrapper around Polygon's Daily Open Close API. "
        "This tool is useful for fetching the daily open/close (stocks) for a ticker. "
        "Input should be the ticker and date"
        "that you want to get the daily open/close for."
    )
    args_schema: Type[PolygonDailyOpenCloseSchema] = PolygonDailyOpenCloseSchema
    api_wrapper: FinancePolygonAPIWrapper

    def _run(
        self,
        stocksTicker: str,
        date: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Polygon API tool."""
        return self.api_wrapper.run(
            mode=self.mode,
            stocksTicker=stocksTicker,
            date=date,
        )
