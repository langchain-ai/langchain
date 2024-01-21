"""Tool for the Google Lens"""

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from langchain_community.utilities.google_lens import GoogleLensAPIWrapper


class GoogleLensQueryRunToolInput(BaseModel):
    query: str = Field(description="Google lens query to search with google lens")


class GoogleLensQueryRun(BaseTool):
    """Tool that queries the Google Lens API."""

    name: str = "google_Lens"
    description: str = (
        "A wrapper around Google Lens Search. "
        "Useful for when you need to get information related"
        "to an image from Google Lens"
        "Input should be a url to an image."
    )
    api_wrapper: GoogleLensAPIWrapper
    args_schema: Type[GoogleLensQueryRunToolInput] = GoogleLensQueryRunToolInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)
