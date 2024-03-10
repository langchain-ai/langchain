from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import BaseTool

from langchain_community.utilities.polygon import PolygonAPIWrapper


class Inputs(BaseModel):
    """Inputs for Polygon's Ticker News API"""

    query: str


class PolygonTickerNews(BaseTool):
    """Tool that gets the latest news for a given ticker from Polygon"""

    mode: str = "get_ticker_news"
    name: str = "polygon_ticker_news"
    description: str = (
        "A wrapper around Polygon's Ticker News API. "
        "This tool is useful for fetching the latest news for a stock. "
        "Input should be the ticker that you want to get the latest news for."
    )
    args_schema: Type[BaseModel] = Inputs

    api_wrapper: PolygonAPIWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Polygon API tool."""
        return self.api_wrapper.run(self.mode, ticker=query)
