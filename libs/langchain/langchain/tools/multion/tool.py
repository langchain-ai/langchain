"""Tool for MultiOn Extension API"""
from typing import Any, Optional

from pydantic import Field

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.base import BaseTool
from langchain.utilities.multion import MultionClientAPIWrapper


def _get_default_multion_client() -> MultionClientAPIWrapper:
    return MultionClientAPIWrapper()


class MultionClientTool(BaseTool):
    """Simulates a Browser interacting agent."""

    name = "Multion_Client"
    description = (
        "A api to communicate with browser extension multion "
        "Useful for automating tasks and actions in the browser "
        "Input should be a task and a url."
        "The result is text form of action that was executed in the given url."
    )
    api_wrapper: MultionClientAPIWrapper = Field(
        default_factory=_get_default_multion_client
    )

    def _run(
        self,
        task: str,
        url: str = "https://www.google.com/",
        tabId: Optional[Any] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.api_wrapper.run(task, url, tabId)
