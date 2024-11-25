from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.finance_polygon import FinancePolygonAPIWrapper


class FinancePolygonReferenceTickerNewsSchema(BaseModel):
    """Input for PolygonReferenceTickerNews."""

    ticker: str = Field(
        description="specify a case-sensitive ticker symbol",
    )
    published_utc: str = Field(
        description="return results published on, before, or after this date.",
    )
    order: str = Field(description="order results based on sort field")
    limit: int = Field(description="limit number of returned results")
    sort: str = Field(description="sort field for ordering")


class PolygonReferenceTickerNews(BaseTool):  # type: ignore[override, override]
    """
    Get the most recent news articles relating to a stock ticker symbol,
    including a summary of the article and a link to the original source.
    """

    mode: str = "get_reference_ticker_news"
    name: str = "polygon_reference_ticker_news"
    description: str = (
        "A wrapper around Polygon's Reference Tickers API. "
        "This tool is useful for fetching news articles for a ticker."
        "Input should be the id, types."
    )
    args_schema: Type[FinancePolygonReferenceTickerNewsSchema] = (
        FinancePolygonReferenceTickerNewsSchema
    )

    api_wrapper: FinancePolygonAPIWrapper

    def _run(
        self,
        ticker: str,
        published_utc: str,
        order: str,
        limit: int,
        sort: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Polygon API tool."""
        return self.api_wrapper.run(
            mode=self.mode,
            ticker=ticker,
            published_utc=published_utc,
            order=order,
            limit=limit,
            sort=sort,
        )
