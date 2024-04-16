"""Tool for the Google Finance"""

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from langchain_community.utilities.google_finance import GoogleFinanceAPIWrapper


class GoogleFinanceQueryRunInput(BaseModel):
    """Input for the GoogleFinanceQueryRun tool."""

    query: str = Field(description="Query for Google Finance")


class GoogleFinanceQueryRun(BaseTool):
    """Tool that queries the Google Finance API."""

    name: str = "google_finance"
    description: str = (
        "A wrapper around Google Finance Search. "
        "Useful for when you need to get information about"
        "google search Finance from Google Finance"
        "Input should be a search query."
    )
    api_wrapper: GoogleFinanceAPIWrapper
    args_schema: Type[BaseModel] = GoogleFinanceQueryRunInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)
