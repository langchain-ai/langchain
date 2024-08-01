from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from langchain_community.utilities.polygon import PolygonAPIWrapper


class Inputs(BaseModel):
    """Inputs for Polygon's Last Quote API"""

    query: str


class PolygonLastQuote(BaseTool):
    """Tool that gets the last quote of a ticker from Polygon"""

    mode: str = "get_last_quote"
    name: str = "polygon_last_quote"
    description: str = (
        "A wrapper around Polygon's Last Quote API. "
        "This tool is useful for fetching the latest price of a stock. "
        "Input should be the ticker that you want to query the last price quote for."
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
