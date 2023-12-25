"""Tool for the TinyImage API."""

from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

from langchain_community.utilities.tiny_image import TinyImageAPIWrapper


class TinyImageRun(BaseTool):
    """Tool that compress image."""

    name: str = "tiny_image"
    description: str = (
        "A wrapper around TinyImage. "
        "Useful for when you need to compress and optimize WebP, JPEG and PNG images."
        "Input should be a image url."
    )
    api_wrapper: TinyImageAPIWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the TinyImage tool."""
        return self.api_wrapper.run(query)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the TinyImage tool."""
        return await self.api_wrapper.arun(query)
