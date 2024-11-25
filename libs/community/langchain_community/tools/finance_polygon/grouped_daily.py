from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.finance_polygon import FinancePolygonAPIWrapper


class PolygonGroupedDailySchema(BaseModel):
    """Inputs for PolygonGroupedDaily."""

    date: str = Field(
        description="The beginning for the aggregate window" "with format YYYY-MM-DD",
    )


class PolygonGroupedDaily(BaseTool):  # type: ignore[override, override]
    """
    Tool that gets the daily open, high, low, and close
    (OHLC) for the entire stocks/equities markets

    """

    mode: str = "get_grouped_daily"
    name: str = "polygon_grouped_daily"
    description: str = (
        "A wrapper around Polygon's Grouped Daily API. "
        "This tool is useful for fetching the daily open, high, low,"
        "and close (OHLC) for the entire stocks/equities markets."
        "Input should be the beginning date of the aggregate window"
    )
    args_schema: Type[PolygonGroupedDailySchema] = PolygonGroupedDailySchema
    api_wrapper: FinancePolygonAPIWrapper

    def _run(
        self,
        date: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Polygon API tool."""
        return self.api_wrapper.run(
            mode=self.mode,
            date=date,
        )
