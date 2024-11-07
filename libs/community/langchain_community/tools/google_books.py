"""Tool for the Google Books API."""

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.google_books import GoogleBooksAPIWrapper


class GoogleBooksQueryInput(BaseModel):
    """Input for the GoogleBooksQuery tool."""

    query: str = Field(description="query to look up on google books")


class GoogleBooksQueryRun(BaseTool):  # type: ignore[override]
    """Tool that searches the Google Books API."""

    name: str = "GoogleBooks"
    description: str = (
        "A wrapper around Google Books. "
        "Useful for when you need to answer general inquiries about "
        "books of certain topics and generate recommendation based "
        "off of key words"
        "Input should be a query string"
    )
    api_wrapper: GoogleBooksAPIWrapper
    args_schema: Type[BaseModel] = GoogleBooksQueryInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Google Books tool."""
        return self.api_wrapper.run(query)
