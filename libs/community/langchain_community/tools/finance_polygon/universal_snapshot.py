from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.finance_polygon import FinancePolygonAPIWrapper


class FinancePolygonUniversalSnapshotSchema(BaseModel):
    """Input for Polygon Universal Snapshot."""

    tickers: str = Field(
        default="",
        description="""
        Comma-separated list of tickers to query up to 250. 
        Empty to return all results.
        """,
    )

    limit: int = Field(
        default=10,
        description=(
            "Limit the number of results returned." "Default is 10 and max is 250."
        ),
    )


class PolygonUniversalSnapshot(BaseTool):  # type: ignore[override, override]
    """
    Tool for fetching snapshots of multiple asset types.
    """

    mode: str = "get_universal_snapshot"
    name: str = "polygon_universal_snapshot"
    description: str = (
        "A wrapper around Polygon's Universal Snapshot API. "
        "This tool is useful for fetching snapshots of assets across various types."
    )

    args_schema: Type[FinancePolygonUniversalSnapshotSchema] = (
        FinancePolygonUniversalSnapshotSchema
    )
    api_wrapper: FinancePolygonAPIWrapper

    def _run(
        self,
        tickers: str,
        limit: int,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Polygon API tool."""
        return self.api_wrapper.run(
            mode=self.mode,
            tickers=tickers,
            limit=limit,
            run_manager=run_manager,
        )
