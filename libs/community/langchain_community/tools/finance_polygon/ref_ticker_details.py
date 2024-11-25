from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.finance_polygon import FinancePolygonAPIWrapper


class FinancePolygonReferenceTickerDetailsSchema(BaseModel):
    """Input for PolygonReferenceTicker."""

    ticker: str = Field(
        description="Specify a ticker symbol.",
    )
    date: str = Field(
        description="time of retreiving ticker on given date",
    )


class PolygonReferenceTickerDetails(BaseTool):  # type: ignore[override, override]
    """
    Tool that gets a single ticker supported by Polygon.io. This response
    will have detailed information about the ticker and the company behind it.
    """

    mode: str = "get_reference_ticker_details"
    name: str = "polygon_reference_ticker_details"
    description: str = (
        "A wrapper around Polygon's Reference Tickers API. "
        "This tool is useful for fetching a details for a ticker symbol."
        "Input should be the ticker, date."
    )
    args_schema: Type[FinancePolygonReferenceTickerDetailsSchema] = (
        FinancePolygonReferenceTickerDetailsSchema
    )

    api_wrapper: FinancePolygonAPIWrapper

    def _run(
        self,
        ticker: str,
        date: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Polygon API tool."""
        return self.api_wrapper.run(mode=self.mode, ticker=ticker, date=date)
