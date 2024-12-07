from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.finance_polygon import FinancePolygonAPIWrapper


class FinancePolygonReferenceTickerEventsSchema(BaseModel):
    """Input for PolygonReferenceTickerEvents."""

    event_id: str = Field(
        description="identifier of an asset.",
    )
    types: str = Field(
        description="comma separated list of the types of event",
    )


class PolygonReferenceTickerEvents(BaseTool):  # type: ignore[override, override]
    """
    Tool that gets a timeline of events for the entity associated
    with the given ticker, CUSIP, or Composite FIGI.
    """

    mode: str = "get_reference_ticker_events"
    name: str = "polygon_reference_ticker_events"
    description: str = (
        "A wrapper around Polygon's Reference Tickers API. "
        "This tool is useful for fetching a timeline of events."
        "Input should be the id, types."
    )
    args_schema: Type[FinancePolygonReferenceTickerEventsSchema] = (
        FinancePolygonReferenceTickerEventsSchema
    )

    api_wrapper: FinancePolygonAPIWrapper

    def _run(
        self,
        event_id: str,
        types: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Polygon API tool."""
        return self.api_wrapper.run(mode=self.mode, event_id=event_id, types=types)
