"""Tool for the Google Books"""

from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

from langchain_community.utilities.google_books import GoogleBooksAPIWrapper


class GoogleBooksQueryRun(BaseTool):
    """Tool that queries the Google book API."""

    name: str = "google_scholar"
    description: str = (
        "A wrapper around Google book Search. "
        "Useful for when you need to get information about"
        "books from Google Books"
        "Input should be a search query."
    )
    api_wrapper: GoogleBooksAPIWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)
