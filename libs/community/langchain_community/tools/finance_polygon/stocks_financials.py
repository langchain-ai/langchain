from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.finance_polygon import FinancePolygonAPIWrapper


class FinancePolygonStocksFinancialsSchema(BaseModel):
    """Input for FinancePolygonStocksFinancials."""

    ticker: str = Field(description="The ticker symbol to fetch financial data for.")
    cik: str = Field(description="The Central Index Key (CIK) for the company.")
    company_name: str = Field(description="The name of the company.")
    sic: str = Field(description="The Standard Industrial Classification (SIC) code.")
    filing_date: str = Field(
        description="The date of the filing." "A date with the format YYYY-MM-DD."
    )
    period_of_report_date: str = Field(
        description="The period of the report for the filing with "
        "financias data. A date with the format YYYY-MM-DD."
    )
    timeframe: str = Field(
        description="The timeframe for the financial data."
        "Annual financials originate from 10-K filings, "
        "quarterly financials originate from 10-Q filings."
        "Possible values are: annual, quartrely, ttm."
    )
    include_sources: str = Field(
        description="Whether or not to include the xbpath and formula "
        "attributes for each finanical data point."
        "Possible values are: true, false."
    )
    order: str = Field(
        description="Order the results on the sort field."
        "Possible values are: asc, desc."
    )
    limit: int = Field(
        description="The number of results to return." "Default is 10 and max is 1000."
    )
    sort: str = Field(
        description="The field to sort by."
        "Possible values are: filing_date, period_of_report_date."
    )


class PolygonStocksFinancials(BaseTool):  # type: ignore[override, override]
    """
    Tool that provides a historical financial data for stock ticker.
    """

    mode: str = "get_stocks_financials"
    name: str = "polygon_stocks_financials"
    description: str = (
        "A wrapper around Polygon's Stocks Financials API. "
        "This tool is useful for fetching historical financial data "
        "for a stock ticker."
    )

    args_schema: Type[FinancePolygonStocksFinancialsSchema] = (
        FinancePolygonStocksFinancialsSchema
    )
    api_wrapper: FinancePolygonAPIWrapper

    def _run(
        self,
        ticker: str,
        cik: str,
        company_name: str,
        sic: str,
        filing_date: str,
        period_of_report_date: str,
        timeframe: str,
        include_sources: str,
        order: str,
        limit: int,
        sort: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Polygon API tool."""
        return self.api_wrapper.run(
            mode=self.mode,
            ticker=ticker,
            cik=cik,
            company_name=company_name,
            sic=sic,
            filing_date=filing_date,
            period_of_report_date=period_of_report_date,
            timeframe=timeframe,
            include_sources=include_sources,
            order=order,
            limit=limit,
            sort=sort,
            run_manager=run_manager,
        )
