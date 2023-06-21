"""Tool for the Google search API."""

from typing import Optional, Type
from pydantic import BaseModel, Field


from langchain.tools.base import BaseTool

from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun

from langchain.utilities.google_appbuilder_api import GoogleAppBuilderAPIWrapper


class GoogleAppBuilderSchema(BaseModel):
    query: str = Field(..., description="Query for google app builder")


class GoogleAppBuilderTool(BaseTool):
    """Tool that adds the capability to query the Google appbuilders API."""

    name = "google_appbuilder"
    description = (
        "A wrapper around Google appbuilders. "
        "Useful for when you need to validate or "
        "discover addressed from ambiguous text. "
        "Input should be a search query."
    )
    # Field(default_factory=GoogleAppBuilderAPIWrapper)
    api_wrapper: GoogleAppBuilderAPIWrapper = GoogleAppBuilderAPIWrapper()
    args_schema: Type[BaseModel] = GoogleAppBuilderSchema

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("GoogleAppBuilderRun does not support async")
