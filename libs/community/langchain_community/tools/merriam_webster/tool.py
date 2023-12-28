"""Tool for the Merriam-Webster API."""

from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

from langchain_community.utilities.merriam_webster import MerriamWebsterAPIWrapper


class MerriamWebsterQueryRun(BaseTool):
    """Tool that searches the Merriam-Webster API."""

    name: str = "MerriamWebster"
    description: str = (
        "A wrapper around Merriam-Webster. "
        "Useful for when you need to get the definition of a word."
        "Input should be the word you want the definition of."
    )
    api_wrapper: MerriamWebsterAPIWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Merriam-Webster tool."""
        return self.api_wrapper.run(query)
