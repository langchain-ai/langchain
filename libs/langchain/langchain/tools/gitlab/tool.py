"""
This tool allows agents to interact with the python-gitlab library
and operate on a GitLab repository.

To use this tool, you must first set as environment variables:
    GITLAB_PRIVATE_ACCESS_TOKEN
    GITLAB_REPOSITORY -> format: {owner}/{repo}

"""
from typing import Optional

from langchain_core.pydantic_v1 import Field

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.base import BaseTool
from langchain.utilities.gitlab import GitLabAPIWrapper


class GitLabAction(BaseTool):
    """Tool for interacting with the GitLab API."""

    api_wrapper: GitLabAPIWrapper = Field(default_factory=GitLabAPIWrapper)
    mode: str
    name: str = ""
    description: str = ""

    def _run(
        self,
        instructions: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the GitLab API to run an operation."""
        return self.api_wrapper.run(self.mode, instructions)
