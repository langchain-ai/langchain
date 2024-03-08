"""
This tool allows agents to interact with the pygithub library
and operate on a GitHub repository.

To use this tool, you must first set as environment variables:
    GITHUB_API_TOKEN
    GITHUB_REPOSITORY -> format: {owner}/{repo}

"""
from typing import Any, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from langchain_community.utilities.github import GitHubAPIWrapper


class GitHubAction(BaseTool):
    """Tool for interacting with the GitHub API."""

    api_wrapper: GitHubAPIWrapper = Field(default_factory=GitHubAPIWrapper)
    mode: str
    name: str = ""
    description: str = ""
    args_schema: Optional[Type[BaseModel]] = None

    def _run(
        self, run_manager: Optional[CallbackManagerForToolRun] = None, **kwargs: Any
    ) -> str:
        """Use the GitHub API to run an operation."""
        if not kwargs or kwargs == {}:
            # Catch other forms of empty input that GPT-4 likes to send.
            return self.api_wrapper.run(self.mode, "")
        return self.api_wrapper.run(self.mode, list(kwargs.values())[0])
