from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.finance_polygon import FinancePolygonAPIWrapper


class FinancePolygonIPOsSchema(BaseModel):
    """Input for PolygonIPOs."""

    ticker: str = Field(
        description="The ticker symbol to fetch aggregates for.",
    )
    us_code: str = Field(
        description="The US code for the company.",
    )
    isin: str = Field(
        description="The ISIN (International Securities Identification "
        "Number for the company.",
    )
    listing_date: str = Field(
        description="The first trading date for the newly listed entity. "
        "A date with the format YYYY-MM-DD."
    )
    ipo_status: str = Field(
        description="The status of the IPO."
        "Possible values are: direct_listing_process, history, "
        "new, pending, postponed, rumor, withdrawn."
    )
    order: str = Field(
        description="The order of the IPO." "Possible values are: asc, desc."
    )
    limit: int = Field(
        description="The number of results to return." "Default is 10 and max is 1000."
    )
    sort: str = Field(
        description="The field to sort by."
        "Possible values are: listing_date, ticker, last_updated, "
        "security_type, issue_name, currency_code, isin, us_code, "
        "final_issue_price, min_shares_offered, max_shares_offered, "
        "lowest_offer_proce, highest_offer_price, total_offer_size, "
        "shares_outstanding, primary_exchange, lot_size, "
        "security_description, ipo_status."
    )


class PolygonIPOs(BaseTool):  # type: ignore[override, override]
    """
    Tool that provides detailed information abotu IPOs including both upcoming
    and historical events.
    """

    mode: str = "get_ipos"
    name: str = "polygon_ipos"
    description: str = (
        "A wrapper around Polygon's IPOs API. "
        "This tool is useful for fetching detailed information"
        " about IPOs for a ticker. "
        "Input should be the ticker, US code, ISIN, listing date, "
        "IPO status, order, limit, and sort."
    )

    args_schema: Type[FinancePolygonIPOsSchema] = FinancePolygonIPOsSchema
    api_wrapper: FinancePolygonAPIWrapper

    def _run(
        self,
        ticker: str,
        us_code: str,
        isin: str,
        listing_date: str,
        ipo_status: str,
        order: str,
        limit: int,
        sort: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Polygon API tool."""
        return self.api_wrapper.run(
            mode=self.mode,
            ticker=ticker,
            us_code=us_code,
            isin=isin,
            listing_date=listing_date,
            ipo_status=ipo_status,
            order=order,
            limit=limit,
            sort=sort,
            run_manager=run_manager,
        )
