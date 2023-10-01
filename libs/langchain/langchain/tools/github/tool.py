"""
This tool allows agents to interact with the pygithub library
and operate on a GitHub repository.

To use this tool, you must first set as environment variables:
    GITHUB_API_TOKEN
    GITHUB_REPOSITORY -> format: {owner}/{repo}

"""
from typing import Optional, Type

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.pydantic_v1 import BaseModel, Field, root_validator
from langchain.tools.base import BaseTool
from langchain.utilities.github import GitHubAPIWrapper


class GitHubToolInput(BaseModel):
    """Goal: have no args{} field"""
    url: str = Field(description='')


class GitHubAction(BaseTool):
    """Tool for interacting with the GitHub API."""

    api_wrapper: GitHubAPIWrapper = Field(default_factory=GitHubAPIWrapper)
    mode: str
    name: str = ""
    description: str = ""
    # args_schema: Type[BaseModel] = GitHubToolInput

    def _run(
        self,
        instructions: Optional[str],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the GitHub API to run an operation."""
        return self.api_wrapper.run(self.mode, instructions)

    @property
    def args(self) -> dict:
        return None
        # TODO: Somehow infer the schema from each inner function!
        # schema = create_schema_from_function(self.name, self.self.api_wrapper.run(self.mode, self.instructions))
        # return schema.schema()["properties"]
