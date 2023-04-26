"""Tool for the GitHub API."""

from langchain.tools.base import BaseTool
from langchain.utilities import GitHubAPIWrapper


class GitHubRepoQueryRun(BaseTool):
    """Tool that adds the capability to query using the GitHub API."""

    api_wrapper: GitHubAPIWrapper

    name = "GitHub"
    description = (
        "A wrapper around GitHub API. "
        "Useful for fetching repository information for a specified user. "
        "Input should be a GitHub username (e.g. 'octocat')."
    )

    def __init__(self) -> None:
        self.api_wrapper = GitHubAPIWrapper()
        return

    def _run(self, user: str) -> str:
        """Use the GitHub tool."""
        return self.api_wrapper.run(user)

    async def _arun(self, user: str) -> str:
        """Use the GitHub tool asynchronously."""
        raise NotImplementedError("GitHubRepoQueryRun does not support async")
