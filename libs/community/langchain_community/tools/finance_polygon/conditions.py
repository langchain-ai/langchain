from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.finance_polygon import FinancePolygonAPIWrapper


class FinancePolygonConditionsSchema(BaseModel):
    """Input for PolygonConditions."""

    asset_class: str = Field(
        description="Filter for conditions within a given asset class."
        "Possible values are: stocks, options, crypto, and fx."
        "Default is stocks."
    )

    data_type: str = Field(
        description="Filter for conditions within a given data type."
        "Possible values are: trade, bbo, nbbo."
    )

    id: str = Field(description="Filter for conditions with a given ID.")

    sip: str = Field(
        description="Filter for conditions with a given SIP."
        "If the condition contains a mapping for that SIP, the condition "
        "will be returned."
    )

    order: str = Field(
        description="Order results based on the sort field."
        "Possible values are: asc, desc."
    )

    limit: int = Field(
        description="The number of results to return." "Default is 10 and max is 1000."
    )

    sort: str = Field(
        description="Sort field used for ordering."
        "Possible values are: asset_class, id, type, name, "
        "data_types, and legacy."
    )


class PolygonConditions(BaseTool):  # type: ignore[override, override]
    """
    Tool that lists all the conditions that Polygon.io uses.
    """

    mode: str = "get_conditions"
    name: str = "polygon_conditions"
    description: str = (
        "A wrapper around Polygon's Conditions API. "
        "This tool is useful for fetching detailed information"
        "about the conditions that Polygon.io uses."
    )

    args_schema: Type[FinancePolygonConditionsSchema] = FinancePolygonConditionsSchema
    api_wrapper: FinancePolygonAPIWrapper

    def _run(
        self,
        asset_class: str,
        data_type: str,
        id: str,
        sip: str,
        order: str,
        limit: int,
        sort: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Polygon API tool."""
        return self.api_wrapper.run(
            mode=self.mode,
            asset_class=asset_class,
            data_type=data_type,
            id=id,
            sip=sip,
            order=order,
            limit=limit,
            sort=sort,
            run_manager=run_manager,
        )
