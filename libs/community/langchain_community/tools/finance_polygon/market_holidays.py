from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

from langchain_community.utilities.finance_polygon import FinancePolygonAPIWrapper


class PolygonMarketHolidays(BaseTool):  # type: ignore[override, override]
    """
    Tool that provides the upcoming market holidays and their
    open/close times.
    """

    mode: str = "get_market_holidays"
    name: str = "polygon_market_holidays"
    description: str = (
        "A wrapper around Polygon's Market Holidays API. "
        "This tool provides the upcoming market holidays and their "
        "open/close times."
    )

    api_wrapper: FinancePolygonAPIWrapper

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Polygon API tool."""
        return self.api_wrapper.run(mode=self.mode, run_manager=run_manager)
