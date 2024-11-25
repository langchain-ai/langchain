from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.finance_polygon import FinancePolygonAPIWrapper


class FinancePolygonCryptoAggregatesSchema(BaseModel):
    """Input for PolygonCryptoAggregates."""

    ticker: str = Field(
        description="The ticker symbol to fetch aggregates for.",
    )
    timespan: str = Field(
        description="The size of the time window. "
        "Possible values are: "
        "second, minute, hour, day, week, month, quarter, year. "
        "Default is 'day'",
    )
    timespan_multiplier: int = Field(
        description="The number of timespans to aggregate. "
        "For example, if timespan is 'day' and "
        "timespan_multiplier is 1, the result will be daily bars. "
        "If timespan is 'day' and timespan_multiplier is 5, "
        "the result will be weekly bars.  "
        "Default is 1.",
    )
    from_date: str = Field(
        description="The start of the aggregate time window. "
        "Either a date with the format YYYY-MM-DD or "
        "a millisecond timestamp.",
    )
    to_date: str = Field(
        description="The end of the aggregate time window. "
        "Either a date with the format YYYY-MM-DD or "
        "a millisecond timestamp.",
    )


class PolygonCryptoAggregates(BaseTool):  # type: ignore[override, override]
    """
    Tool that gets aggregate bars (cryptocurrencies) over a
    given date range for a given ticker from Polygon.
    """

    mode: str = "get_crypto_aggregates"
    name: str = "polygon_crypto_aggregates"
    description: str = (
        "A wrapper around Polygon's Crypto Aggregates API. "
        "This tool is useful for fetching aggregate bars for a ticker."
        "Input should be the ticker, date range, timespan, and timespan multiplier"
        " that you want to get the aggregate bars for."
    )
    args_schema: Type[FinancePolygonCryptoAggregatesSchema] = (
        FinancePolygonCryptoAggregatesSchema
    )

    api_wrapper: FinancePolygonAPIWrapper

    def _run(
        self,
        ticker: str,
        timespan: str,
        timespan_multiplier: int,
        from_date: str,
        to_date: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Polygon API tool."""
        return self.api_wrapper.run(
            mode=self.mode,
            ticker=ticker,
            timespan=timespan,
            timespan_multiplier=timespan_multiplier,
            from_date=from_date,
            to_date=to_date,
        )
