from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.finance_polygon import FinancePolygonAPIWrapper


class FinancePolygonRelatedCompaniesSchema(BaseModel):
    """Input for PolygonRelatedCompanies."""

    ticker: str = Field(
        description="The ticker symbol to fetch related companies for.",
    )


class PolygonRelatedCompanies(BaseTool):  # type: ignore[override, override]
    """
    Tool that provides detailed information about related companies for a given ticker.
    """

    mode: str = "get_related_companies"
    name: str = "polygon_related_companies"
    description: str = (
        "A wrapper around Polygon's Related Companies API. "
        "This tool is useful for fetching detailed information about related companies"
        " for a given ticker."
    )

    args_schema: Type[FinancePolygonRelatedCompaniesSchema] = (
        FinancePolygonRelatedCompaniesSchema
    )

    api_wrapper: FinancePolygonAPIWrapper

    def _run(
        self,
        ticker: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        Get related companies for a given ticker.
        """
        return self.api_wrapper.run(
            mode=self.mode, ticker=ticker, run_manager=run_manager
        )
