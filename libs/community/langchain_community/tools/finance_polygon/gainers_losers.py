from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.finance_polygon import FinancePolygonAPIWrapper


class FinancePolygonGainersLosersSchema(BaseModel):
    """Input for Polygon Top Gainers/Losers."""

    direction: str = Field(
        description="Direction of the snapshot results."
        "Possible values are 'gainers' or 'losers'."
    )

    include_otc: bool = Field(
        default=False, description="Include OTC securities. Default is false."
    )


class PolygonGainersLosers(BaseTool):  # type: ignore[override, override]
    """
    Tool for fetching top gainers or losers in the market.
    """

    mode: str = "get_top_gainers_losers"
    name: str = "polygon_gainers_losers"
    description: str = (
        "A wrapper around Polygon's Gainers/Losers API. "
        "This tool is useful for fetching the top gainers or losers of the day."
    )

    args_schema: Type[FinancePolygonGainersLosersSchema] = (
        FinancePolygonGainersLosersSchema
    )
    api_wrapper: FinancePolygonAPIWrapper

    def _run(
        self,
        direction: str,
        include_otc: bool,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Polygon API tool."""
        return self.api_wrapper.run(
            mode=self.mode,
            direction=direction,
            include_otc=include_otc,
            run_manager=run_manager,
        )
