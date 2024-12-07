from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.finance_polygon import FinancePolygonAPIWrapper


class PolygonDividendsSchema(BaseModel):
    """Inputs for PolygonDividends."""

    ticker: str = Field(
        description="The ticker symbol to fetch dividends for.",
    )
    ex_dividend_date: str = Field(
        description="The ex-dividend date to query by," "with format YYYY-MM-DD",
    )
    record_date: str = Field(
        description="The record date to query by," "with format YYYY-MM-DD",
    )
    declaration_date: str = Field(
        description="The declaration date to query by," "with format YYYY-MM-DD",
    )
    pay_date: str = Field(
        description="The pay date to query by," "with format YYYY-MM-DD",
    )
    frequency: int = Field(
        description="Query by the number of times the dividend is paid out per year"
        "Possible values are are 0 (one-time), 1 (annually), 2 (bi-annually),"
        "4 (quarterly), and 12 (monthly)."
    )
    cash_amount: int = Field(description="The cash amount of the dividend to query by")
    dividend_type: str = Field(
        description="Type of dividend to query by."
        "Possible options are CD, SD, ST, or LT"
    )


class PolygonDividends(BaseTool):  # type: ignore[override, override]
    """
    Tool that gets a list of historical cash dividends,
    including the ticker symbol, declaration date, ex-dividend date,
    record date, pay date, frequency, and amount from Polygon
    """

    mode: str = "get_dividenda"
    name: str = "polygon_dividends"
    description: str = "later"
    args_schema: Type[PolygonDividendsSchema] = PolygonDividendsSchema
    api_wrapper: FinancePolygonAPIWrapper

    def _run(
        self,
        ticker: str,
        ex_dividend_date: str,
        record_date: str,
        declaration_date: str,
        pay_date: str,
        frequency: int,
        cash_amount: int,
        dividend_type: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Polygon API tool."""
        return self.api_wrapper.run(
            mode=self.mode,
            ticker=ticker,
            ex_dividend_date=ex_dividend_date,
            record_date=record_date,
            declaration_date=declaration_date,
            pay_date=pay_date,
            frequency=frequency,
            cash_amount=cash_amount,
            dividend_type=dividend_type,
        )
