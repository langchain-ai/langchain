from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from langchain_community.utilities.financial_datasets import FinancialDatasetsAPIWrapper


class IncomeStatementsSchema(BaseModel):
    """Input for IncomeStatements."""

    ticker: str = Field(
        description="The ticker symbol to fetch income statements for.",
    )
    period: str = Field(
        description="The period of the income statement. "
        "Possible values are: "
        "annual, quarterly, ttm. "
        "Default is 'annual'.",
    )
    limit: int = Field(
        description="The number of income statements to return. " "Default is 10.",
    )


class IncomeStatements(BaseTool):
    """
    Tool that gets income statements for a given ticker over a given period.
    """

    mode: str = "get_income_statements"
    name: str = "income_statements"
    description: str = (
        "A wrapper around financial datasets's Income Statements API. "
        "This tool is useful for fetching income statements for a given ticker."
        "The tool fetches income statements for a given ticker over a given period."
        "The period can be annual, quarterly, or trailing twelve months (ttm)."
        "The number of income statements to return can also be "
        "specified using the limit parameter."
    )
    args_schema: Type[IncomeStatementsSchema] = IncomeStatementsSchema

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
        """Use the Income Statements API tool."""
        return self.api_wrapper.run(
            mode=self.mode,
            ticker=ticker,
            period=period,
            limit=limit,
        )
