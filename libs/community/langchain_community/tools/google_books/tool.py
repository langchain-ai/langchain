"""Tool for the Google Books API."""

from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

class GoogleBooksQueryRun(BaseTool):
    """Tool that searches the Google Books API."""

    name: str = "GoogleBooks"
    description: str = (
        "A wrapper around Google Books. "
    )
    api_wrapper: GoogleBooksApiWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Google Books tool."""
        return self.api_wrapper.run(query)