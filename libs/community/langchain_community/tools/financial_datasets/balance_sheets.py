from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.financial_datasets import FinancialDatasetsAPIWrapper


class BalanceSheetsSchema(BaseModel):
    """Input for BalanceSheets."""

    ticker: str = Field(
        description="The ticker symbol to fetch balance sheets for.",
    )
    period: str = Field(
        description="The period of the balance sheets. "
        "Possible values are: "
        "annual, quarterly, ttm. "
        "Default is 'annual'.",
    )
    limit: int = Field(
        description="The number of balance sheets to return. Default is 10.",
    )


class BalanceSheets(BaseTool):
    """
    Tool that gets balance sheets for a given ticker over a given period.
    """

    mode: str = "get_balance_sheets"
    name: str = "balance_sheets"
    description: str = (
        "A wrapper around financial datasets's Balance Sheets API. "
        "This tool is useful for fetching balance sheets for a given ticker."
        "The tool fetches balance sheets for a given ticker over a given period."
        "The period can be annual, quarterly, or trailing twelve months (ttm)."
        "The number of balance sheets to return can also be "
        "specified using the limit parameter."
    )
    args_schema: Type[BalanceSheetsSchema] = BalanceSheetsSchema

    api_wrapper: FinancialDatasetsAPIWrapper = Field(..., exclude=True)

    def __init__(self, api_wrapper: FinancialDatasetsAPIWrapper):
        super().__init__(api_wrapper=api_wrapper)

    def _run(
        self,
        ticker: str,
        period: str,
        limit: Optional[int],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Balance Sheets API tool."""
        return self.api_wrapper.run(
            mode=self.mode,
            ticker=ticker,
            period=period,
            limit=limit,
        )
