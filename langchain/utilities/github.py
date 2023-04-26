"""Util that calls GitHub API using requests."""
from typing import Any, Dict, Optional

import requests
from pydantic import Extra, root_validator

from langchain.tools.base import BaseModel
from langchain.utils import get_from_dict_or_env

""" 
Example usage

from langchain.utilities import GitHubAPIWrapper

#set env variable GITHUB_API_KEY
import os

os.environ['GITHUB_API_KEY'] = ''

github_api = GitHubAPIWrapper()

print(github_api.run('SamPink'))
"""


class GitHubAPIWrapper(BaseModel):
    """Wrapper for GitHub API using requests.

    Docs for using:

    1. Go to GitHub and sign up for an API key
    2. Save your API KEY into GITHUB_API_KEY env variable
    3. pip install requests
    """

    github_api_key: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        github_api_key = get_from_dict_or_env(
            values, "github_api_key", "GITHUB_API_KEY"
        )
        values["github_api_key"] = github_api_key

        try:
            import requests

        except ImportError:
            raise ImportError(
                "requests is not installed. "
                "Please install it with `pip install requests`"
            )

        return values

    def _format_repo_info(self, repo: Dict) -> str:
        return (
            f"Repo name: {repo['name']}\n"
            f"Description: {repo['description']}\n"
            f"URL: {repo['html_url']}\n"
            f"Stars: {repo['stargazers_count']}\n"
            f"Forks: {repo['forks_count']}\n"
            f"Language: {repo['language']}\n"
            f"Open issues: {repo['open_issues_count']}\n"
        )

    def run(self, user: str) -> str:
        """Get the repository information for a specified user."""
        headers = {"Authorization": f"token {self.github_api_key}"}
        response = requests.get(
            f"https://api.github.com/users/{user}/repos", headers=headers
        )

        if response.status_code != 200:
            raise ValueError(f"Error: {response.status_code}, {response.text}")

        repos = response.json()
        repo_info = "\n\n".join(self._format_repo_info(repo) for repo in repos)

        return f"Repositories for {user}:\n\n{repo_info}"
