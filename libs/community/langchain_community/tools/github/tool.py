"""
This tool allows agents to interact with the pygithub library
and operate on a GitHub repository.

To use this tool, you must first set as environment variables:
    GITHUB_API_TOKEN
    GITHUB_REPOSITORY -> format: {owner}/{repo}

"""

from typing import Any, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.github import GitHubAPIWrapper


class GitHubAction(BaseTool):
    """Tool for interacting with the GitHub API."""

    api_wrapper: GitHubAPIWrapper = Field(default_factory=GitHubAPIWrapper)
    mode: str
    name: str = ""
    description: str = ""
    args_schema: Optional[Type[BaseModel]] = None

    def _run(
        self,
        instructions: Optional[str] = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> str:
        """Use the GitHub API to run an operation."""
        if not instructions or instructions == "{}":
            # Catch other forms of empty input that GPT-4 likes to send.
            instructions = ""
        if self.args_schema is not None:
            field_names = list(self.args_schema.schema()["properties"].keys())
            if len(field_names) > 1:
                raise AssertionError(
                    f"Expected one argument in tool schema, got {field_names}."
                )
            if field_names:
                field = field_names[0]
            else:
                field = ""
            query = str(kwargs.get(field, ""))
        else:
            query = instructions
        return self.api_wrapper.run(self.mode, query)
