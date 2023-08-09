import logging
from typing import Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools import BaseTool

from ...utilities.google_drive import FORMAT_INSTRUCTION, GoogleDriveAPIWrapper

logger = logging.getLogger(__name__)


class GoogleDriveSearchTool(BaseTool):
    """Tool that adds the capability to query the Google Drive search API."""

    name = "Google Drive Search"
    description = (
        "A wrapper around Google Drive Search. "
        "Useful for when you need to find a document in google drive. "
        f"{FORMAT_INSTRUCTION}"
    )
    api_wrapper: GoogleDriveAPIWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        logger.info(f"{query=}")
        return self.api_wrapper.run(query)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("GoogleSearchRun does not support async")
