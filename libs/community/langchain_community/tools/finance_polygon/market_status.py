from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

from langchain_community.utilities.finance_polygon import FinancePolygonAPIWrapper


class PolygonMarketStatus(BaseTool):  # type: ignore[override, override]
    """
    Tool that provides the current trading status of the exchanges
    and overall financial markets
    """

    mode: str = "get_market_status"
    name: str = "polygon_market_status"
    description: str = (
        "A wrapper around Polygon's Market Status API. "
        "This tool provides the current trading status of the exchanges "
        "and overall financial markets."
    )

    api_wrapper: FinancePolygonAPIWrapper

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Polygon API tool."""
        return self.api_wrapper.run(mode=self.mode, run_manager=run_manager)
