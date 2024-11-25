from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.finance_polygon import FinancePolygonAPIWrapper


class FinancePolygonExchangesSchema(BaseModel):
    """Input for PolygonExchanges."""

    asset_class: str = Field(
        description="Filter by asset class."
        "Possible values are: stocks, options, crypto, and fx."
        "Default is stocks."
    )

    locale: str = Field(
        description="Filter by locale."
        "Possible values are: us, global."
        "Default is us."
    )


class PolygonExchanges(BaseTool):  # type: ignore[override, override]
    """
    Tool that lists all exchances that Polygon.io knows about
    """

    mode: str = "get_exchanges"
    name: str = "polygon_exchanges"
    description: str = (
        "A wrapper around Polygon's Exchanges API. "
        "This tool is useful for fetching detailed information about exchanges."
    )

    args_schema: Type[FinancePolygonExchangesSchema] = FinancePolygonExchangesSchema

    api_wrapper: FinancePolygonAPIWrapper

    def _run(
        self,
        asset_class: str,
        locale: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        Get exchanges for a given asset class and locale.
        """
        return self.api_wrapper.run(
            mode=self.mode,
            asset_class=asset_class,
            locale=locale,
            run_manager=run_manager,
        )
