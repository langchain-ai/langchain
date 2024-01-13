from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

from langchain_community.utilities.polygon import PolygonAPIWrapper


class PolygonLastQuote(BaseTool):
    """Tool that gets the last quote of a ticker from Polygon"""

    mode: str
    name: str = "polygon_last_quote"
    description: str = (
        "A wrapper around Polygon's Last Quote API."
        "This tool is useful for fetching the latest price of a stock."
        "Input should be the ticker that you want to query the last price quote for."
    )

    api_wrapper: PolygonAPIWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Polygon API tool."""
        return self.api_wrapper.run(self.mode, ticker=query)
