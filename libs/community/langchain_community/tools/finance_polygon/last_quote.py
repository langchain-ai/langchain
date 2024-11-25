from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.finance_polygon import FinancePolygonAPIWrapper


class PolygonLastQuoteSchema(BaseModel):
    """Inputs for PolygonLastQuote."""

    ticker: str = Field(
        description="The ticker symbol to fetch the NBBO tick for.",
    )


class FinancePolygonLastQuote(BaseTool):  # type: ignore[override, override]
    """
    Tool that gets the most recent NBBO (Quote) tick for a given stock.
    """

    mode: str = "get_last_quote"
    name: str = "finance_polygon_last_quote"
    description: str = (
        "A wrapper around Polygon's Last Quote API. "
        "This tool is useful for fetching the most recent NBBO (Quote) tick"
        "Input should be the ticker symbol that you want to"
        "get the quote for."
    )
    args_schema: Type[PolygonLastQuoteSchema] = PolygonLastQuoteSchema
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
