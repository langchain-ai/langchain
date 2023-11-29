"""Tool for Steam Web API"""

from typing import Optional

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.base import BaseTool
from langchain.utilities.steam import SteamWebAPIWrapper


class SteamWebAPIQueryRun(BaseTool):
    """Tool that searches the Steam Web API."""

    mode: str
    name: str = "Steam"
    description: str = (
        "A wrapper around Steam Web API."
        "Steam Tool is useful for fetching User profiles and stats, Game data and more!"
        "Input should be the User or Game you want to query."
    )

    api_wrapper: SteamWebAPIWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Steam-WebAPI tool."""
        return self.api_wrapper.run(self.mode, query)
