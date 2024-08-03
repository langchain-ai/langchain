"""
Util that calls several of financial datasets stock market REST APIs.
Docs: https://docs.financialdatasets.ai/
"""

import json
from typing import Any, List, Optional

import requests
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.utils import get_from_dict_or_env

FINANCIAL_DATASETS_BASE_URL = "https://api.financialdatasets.ai/"


class FinancialDatasetsAPIWrapper(BaseModel):
    """Wrapper for financial datasets API."""

    financial_datasets_api_key: Optional[str] = None

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.financial_datasets_api_key = get_from_dict_or_env(
            data, "financial_datasets_api_key", "FINANCIAL_DATASETS_API_KEY"
        )

    def get_income_statements(
        self,
        ticker: str,
        period: str,
        limit: Optional[int],
    ) -> Optional[dict]:
        """
        Get the income statements for a stock `ticker` over a `period` of time.

        :param ticker: the stock ticker
        :param period: the period of time to get the balance sheets for.
            Possible values are: annual, quarterly, ttm.
        :param limit: the number of results to return, default is 10
        :return: a list of income statements
        """
        url = (
            f"{FINANCIAL_DATASETS_BASE_URL}financials/income-statements/"
            f"?ticker={ticker}"
            f"&period={period}"
            f"&limit={limit if limit else 10}"
        )

        # Add the api key to the headers
        headers = {"X-API-KEY": self.financial_datasets_api_key}

        # Execute the request
        response = requests.get(url, headers=headers)
        data = response.json()

        return data.get("income_statements", None)

    def get_balance_sheets(
        self,
        ticker: str,
        period: str,
        limit: Optional[int],
    ) -> List[dict]:
        """
        Get the balance sheets for a stock `ticker` over a `period` of time.

        :param ticker: the stock ticker
        :param period: the period of time to get the balance sheets for.
            Possible values are: annual, quarterly, ttm.
        :param limit: the number of results to return, default is 10
        :return: a list of balance sheets
        """
        url = (
            f"{FINANCIAL_DATASETS_BASE_URL}financials/balance-sheets/"
            f"?ticker={ticker}"
            f"&period={period}"
            f"&limit={limit if limit else 10}"
        )

        # Add the api key to the headers
        headers = {"X-API-KEY": self.financial_datasets_api_key}

        # Execute the request
        response = requests.get(url, headers=headers)
        data = response.json()

        return data.get("balance_sheets", None)

    def get_cash_flow_statements(
        self,
        ticker: str,
        period: str,
        limit: Optional[int],
    ) -> List[dict]:
        """
        Get the cash flow statements for a stock `ticker` over a `period` of time.

        :param ticker: the stock ticker
        :param period: the period of time to get the balance sheets for.
            Possible values are: annual, quarterly, ttm.
        :param limit: the number of results to return, default is 10
        :return: a list of cash flow statements
        """

        url = (
            f"{FINANCIAL_DATASETS_BASE_URL}financials/cash-flow-statements/"
            f"?ticker={ticker}"
            f"&period={period}"
            f"&limit={limit if limit else 10}"
        )

        # Add the api key to the headers
        headers = {"X-API-KEY": self.financial_datasets_api_key}

        # Execute the request
        response = requests.get(url, headers=headers)
        data = response.json()

        return data.get("cash_flow_statements", None)

    def run(self, mode: str, ticker: str, **kwargs: Any) -> str:
        if mode == "get_income_statements":
            period = kwargs.get("period", "annual")
            limit = kwargs.get("limit", 10)
            return json.dumps(self.get_income_statements(ticker, period, limit))
        elif mode == "get_balance_sheets":
            period = kwargs.get("period", "annual")
            limit = kwargs.get("limit", 10)
            return json.dumps(self.get_balance_sheets(ticker, period, limit))
        elif mode == "get_cash_flow_statements":
            period = kwargs.get("period", "annual")
            limit = kwargs.get("limit", 10)
            return json.dumps(self.get_cash_flow_statements(ticker, period, limit))
        else:
            raise ValueError(f"Invalid mode {mode} for financial datasets API.")
