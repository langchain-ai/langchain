"""Tool for the Google search API."""

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from langchain_community.utilities.google_places_api import GooglePlacesAPIWrapper


class GooglePlacesSchema(BaseModel):
    """Input for GooglePlacesTool."""

    query: str = Field(..., description="Query for google maps")


class GooglePlacesTool(BaseTool):
    """Tool that queries the Google places API."""

    name: str = "google_places"
    description: str = (
        "A wrapper around Google Places. "
        "Useful for when you need to validate or "
        "discover addressed from ambiguous text. "
        "Input should be a search query."
    )
    api_wrapper: GooglePlacesAPIWrapper = Field(default_factory=GooglePlacesAPIWrapper)
    args_schema: Type[BaseModel] = GooglePlacesSchema

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)
