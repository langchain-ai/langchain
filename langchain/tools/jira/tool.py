from typing import Any, Dict, Optional

from pydantic import Field, root_validator

from langchain.tools.base import BaseTool
from langchain.utilities.jira import JiraAPIWrapper


class JiraAction(BaseTool):
    """
    Args:
        action_id: a specific action ID (from list actions) of the action to execute
            (the set api_key must be associated with the action owner)
        instructions: a natural language instruction string for using the action
            (eg. "get the latest email from Mike Knoop" for "Gmail: find email" action)
    """

    api_wrapper: JiraAPIWrapper = Field(default_factory=JiraAPIWrapper)
    action_id: str
    name = ""
    description = ""

    def _run(self, instructions: str) -> str:
        """Use the Zapier NLA tool to return a list of all exposed user actions."""
        return self.api_wrapper.run(self.action_id, instructions)

    async def _arun(self, _: str) -> str:
        """Use the Zapier NLA tool to return a list of all exposed user actions."""
        raise NotImplementedError("ZapierNLAListActions does not support async")
