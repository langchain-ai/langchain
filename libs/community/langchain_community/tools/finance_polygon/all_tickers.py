from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.finance_polygon import FinancePolygonAPIWrapper


class FinancePolygonAllTickersSchema(BaseModel):
    """Input for Polygon All Tickers."""

    tickers: str = Field(
        default="",
        description=(
            "A case-sensitive comma-separated list of tickers for snapshot."
            "Empty for all tickers."
        ),
    )

    include_otc: bool = Field(
        default=False, description="Include OTC securities. Default is false."
    )


class PolygonAllTickers(BaseTool):  # type: ignore[override, override]
    """
    Tool for fetching all traded stock symbols with snapshot data.
    """

    mode: str = "get_all_tickers"
    name: str = "polygon_all_tickers"
    description: str = (
        "A wrapper around Polygon's All Tickers API. "
        "This tool provides the most up-to-date market data"
        "for all or specified traded stock symbols."
    )

    args_schema: Type[FinancePolygonAllTickersSchema] = FinancePolygonAllTickersSchema
    api_wrapper: FinancePolygonAPIWrapper

    def _run(
        self,
        tickers: str,
        include_otc: bool,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Polygon API tool."""
        return self.api_wrapper.run(
            mode=self.mode,
            tickers=tickers,
            include_otc=include_otc,
            run_manager=run_manager,
        )
