from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.finance_polygon import FinancePolygonAPIWrapper


class FinancePolygonReferenceTickerSchema(BaseModel):
    """Input for PolygonReferenceTicker."""

    ticker: str = Field(
        description="Specify a ticker symbol.",
    )
    ticker_type: str = Field(
        description="Specify type of ticker",
    )
    market: str = Field(
        description="Filter by market type",
    )
    exchange: str = Field(
        description="specify primary exchange of asset in" "ISO code format",
    )
    cusip: str = Field(
        description="specify CIK of asset",
    )
    date: str = Field(
        description="time of retreiving ticker on given date",
    )
    search: str = Field(
        description="search for terms and/or company name",
    )
    active: str = Field(
        description="specify if returned ticker should be"
        "actively traded on the queried date",
    )
    order: str = Field(
        description="order results based on asc or desc order",
    )
    limit: int = Field(
        description="limit the number of results",
    )
    sort: str = Field(description="sort based on a field")


class PolygonReferenceTicker(BaseTool):  # type: ignore[override, override]
    """
    Tool that queries all ticker symbols supported by Polygon.io.
    Currently includes Stocks,Indices,Forex,Crypto.
    """

    mode: str = "get_reference_tickers"
    name: str = "polygon_reference_tickers"
    description: str = (
        "A wrapper around Polygon's Reference Tickers API. "
        "This tool is useful for fetching all ticker symbols supported by Polygon. "
        "Input should be the ticker, type, market, exchange, cusip, cik"
        "date, search, active, order, limit, sort."
    )
    args_schema: Type[FinancePolygonReferenceTickerSchema] = (
        FinancePolygonReferenceTickerSchema
    )

    api_wrapper: FinancePolygonAPIWrapper

    def _run(
        self,
        ticker: str,
        ticker_type: str,
        market: str,
        exchange: str,
        cusip: str,
        cik: str,
        date: str,
        search: str,
        active: str,
        order: str,
        limit: int,
        sort: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Polygon API tool."""
        return self.api_wrapper.run(
            mode=self.mode,
            ticker=ticker,
            ticker_type=ticker_type,
            market=market,
            exchange=exchange,
            cusip=cusip,
            cik=cik,
            date=date,
            search=search,
            active=active,
            order=order,
            limit=limit,
            sort=sort,
        )
