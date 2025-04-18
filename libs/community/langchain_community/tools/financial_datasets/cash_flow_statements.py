from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.financial_datasets import FinancialDatasetsAPIWrapper


class CashFlowStatementsSchema(BaseModel):
    """Input for CashFlowStatements."""

    ticker: str = Field(
        description="The ticker symbol to fetch cash flow statements for.",
    )
    period: str = Field(
        description="The period of the cash flow statement. "
        "Possible values are: "
        "annual, quarterly, ttm. "
        "Default is 'annual'.",
    )
    limit: int = Field(
        description="The number of cash flow statements to return. Default is 10.",
    )


class CashFlowStatements(BaseTool):
    """
    Tool that gets cash flow statements for a given ticker over a given period.
    """

    mode: str = "get_cash_flow_statements"
    name: str = "cash_flow_statements"
    description: str = (
        "A wrapper around financial datasets's Cash Flow Statements API. "
        "This tool is useful for fetching cash flow statements for a given ticker."
        "The tool fetches cash flow statements for a given ticker over a given period."
        "The period can be annual, quarterly, or trailing twelve months (ttm)."
        "The number of cash flow statements to return can also be "
        "specified using the limit parameter."
    )
    args_schema: Type[CashFlowStatementsSchema] = CashFlowStatementsSchema

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
        """Use the Cash Flow Statements API tool."""
        return self.api_wrapper.run(
            mode=self.mode,
            ticker=ticker,
            period=period,
            limit=limit,
        )
