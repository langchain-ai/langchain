from typing import Any, Dict, Optional

from pydantic import Field, root_validator

from langchain.tools.base import BaseTool
from langchain.utilities.jira import JiraAPIWrapper


class JiraAction(BaseTool):
    api_wrapper: JiraAPIWrapper = Field(default_factory=JiraAPIWrapper)
    action_id: str
    name = ""
    description = ""

    def _run(self, instructions: str) -> str:
        """Use the Atlassian Jira API to run an operation."""
        return self.api_wrapper.run(self.action_id, instructions)

    async def _arun(self, _: str) -> str:
        """Use the Atlassian Jira API to run an operation."""
        raise NotImplementedError("JiraAction does not support async")
