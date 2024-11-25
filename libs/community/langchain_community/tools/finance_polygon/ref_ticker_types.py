from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.finance_polygon import FinancePolygonAPIWrapper


class FinancePolygonReferenceTickerTypesSchema(BaseModel):
    """Input for PolygonReferenceTickerTypes."""

    asset_class: str = Field(
        description="filter by asset.",
    )
    locale: str = Field(
        description="filter by locale",
    )


class PolygonReferenceTickerTypes(BaseTool):  # type: ignore[override, override]
    """
    Tool that lists all ticker types that Polygon.io has.
    """

    mode: str = "get_reference_ticker_types"
    name: str = "polygon_reference_ticker_types"
    description: str = (
        "A wrapper around Polygon's Reference Tickers API. "
        "This tool is useful for fetching all ticker types"
    )
    args_schema: Type[FinancePolygonReferenceTickerTypesSchema] = (
        FinancePolygonReferenceTickerTypesSchema
    )

    api_wrapper: FinancePolygonAPIWrapper

    def _run(
        self,
        asset_class: str,
        locale: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Polygon API tool."""
        return self.api_wrapper.run(
            mode=self.mode, asset_class=asset_class, locale=locale
        )
