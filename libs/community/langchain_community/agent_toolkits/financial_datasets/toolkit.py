from __future__ import annotations

from typing import List

from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool
from langchain_core.tools.base import BaseToolkit

from langchain_community.tools.financial_datasets.balance_sheets import BalanceSheets
from langchain_community.tools.financial_datasets.cash_flow_statements import (
    CashFlowStatements,
)
from langchain_community.tools.financial_datasets.income_statements import (
    IncomeStatements,
)
from langchain_community.utilities.financial_datasets import FinancialDatasetsAPIWrapper


class FinancialDatasetsToolkit(BaseToolkit):
    """Toolkit for interacting with financialdatasets.ai.

    Parameters:
        api_wrapper: The FinancialDatasets API Wrapper.
    """

    api_wrapper: FinancialDatasetsAPIWrapper = Field(
        default_factory=FinancialDatasetsAPIWrapper
    )

    def __init__(self, api_wrapper: FinancialDatasetsAPIWrapper):
        super().__init__()
        self.api_wrapper = api_wrapper

    class Config:
        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            BalanceSheets(api_wrapper=self.api_wrapper),
            CashFlowStatements(api_wrapper=self.api_wrapper),
            IncomeStatements(api_wrapper=self.api_wrapper),
        ]
