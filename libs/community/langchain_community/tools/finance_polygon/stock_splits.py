from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.tools.finance_polygon.stocks_financials import (
    FinancePolygonStocksFinancialsSchema,
)
from langchain_community.utilities.finance_polygon import FinancePolygonAPIWrapper


class FinancePolygonStockSplitsSchema(BaseModel):
    """Input for PolygonStockSplits."""

    ticker: str = Field(
        description="The ticker symbol to fetch information about splits for."
    )

    execution_date: str = Field(
        description="The date the split was executed."
        "A date with the format YYYY-MM-DD."
    )

    reverse_split: str = Field(
        description="Filter for reverse splits."
        "By default, this filter is not used."
        "Possible values are: true, false."
    )

    order: str = Field(
        description="Order the results based on the sort filter."
        "Possible values are: asc, desc."
    )

    limit: int = Field(
        description="The number of results to return." "Default is 10 and max is 1000."
    )

    sort: str = Field(
        description="Sort field used for ordering."
        "Possible values are: execution_date, ticker."
    )


class PolygonStockSplits(BaseTool):  # type: ignore[override, override]
    """
    Tool that provides a list of historical stock splits,
    including ticker symbol, the execution date, and the
    factors of the split ratio.
    """

    mode: str = "get_stock_splits"
    name: str = "polygon_stock_splits"
    description: str = (
        "A wrapper around Polygon's Stock Splits API. "
        "This tool is useful for fetching detailed information"
        "about historical stock splits."
    )

    args_schema: Type[FinancePolygonStocksFinancialsSchema] = (
        FinancePolygonStocksFinancialsSchema
    )
    api_wrapper: FinancePolygonAPIWrapper

    def _run(
        self,
        ticker: str,
        execution_date: str,
        reverse_split: str,
        order: str,
        limit: int,
        sort: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Polygon API tool."""
        return self.api_wrapper.run(
            mode=self.mode,
            ticker=ticker,
            execution_date=execution_date,
            reverse_split=reverse_split,
            order=order,
            limit=limit,
            sort=sort,
            run_manager=run_manager,
        )
