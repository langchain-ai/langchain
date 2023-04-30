"""Tool for the Google search API."""

from pydantic import Field

from langchain.tools.base import BaseTool
from langchain.utilities.google_places_api import GooglePlacesAPIWrapper


class GooglePlacesTool(BaseTool):
    """Tool that adds the capability to query the Google places API."""

    name = "Google Places"
    description = (
        "A wrapper around Google Places. "
        "Useful for when you need to validate or "
        "discover addressed from ambiguous text. "
        "Input should be a search query."
    )
    api_wrapper: GooglePlacesAPIWrapper = Field(default_factory=GooglePlacesAPIWrapper)

    def _run(self, query: str) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("GooglePlacesRun does not support async")
