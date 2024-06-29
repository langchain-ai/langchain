"""GitHub Toolkit."""

from typing import Dict, List

from langchain_core.tools import BaseToolkit

from langchain_community.tools import BaseTool
from langchain_community.tools.gitlab.prompt import (
    COMMENT_ON_ISSUE_PROMPT,
    CREATE_FILE_PROMPT,
    CREATE_PULL_REQUEST_PROMPT,
    DELETE_FILE_PROMPT,
    GET_ISSUE_PROMPT,
    GET_ISSUES_PROMPT,
    READ_FILE_PROMPT,
    UPDATE_FILE_PROMPT,
)
from langchain_community.tools.gitlab.tool import GitLabAction
from langchain_community.utilities.gitlab import GitLabAPIWrapper


class GitLabToolkit(BaseToolkit):
    """GitLab Toolkit.

    *Security Note*: This toolkit contains tools that can read and modify
        the state of a service; e.g., by creating, deleting, or updating,
        reading underlying data.

        For example, this toolkit can be used to create issues, pull requests,
        and comments on GitLab.

        See https://python.langchain.com/docs/security for more information.

    Parameters:
        tools: List[BaseTool]. The tools in the toolkit. Default is an empty list.
    """

    tools: List[BaseTool] = []

    @classmethod
    def from_gitlab_api_wrapper(
        cls, gitlab_api_wrapper: GitLabAPIWrapper
    ) -> "GitLabToolkit":
        operations: List[Dict] = [
            {
                "mode": "get_issues",
                "name": "Get Issues",
                "description": GET_ISSUES_PROMPT,
            },
            {
                "mode": "get_issue",
                "name": "Get Issue",
                "description": GET_ISSUE_PROMPT,
            },
            {
                "mode": "comment_on_issue",
                "name": "Comment on Issue",
                "description": COMMENT_ON_ISSUE_PROMPT,
            },
            {
                "mode": "create_pull_request",
                "name": "Create Pull Request",
                "description": CREATE_PULL_REQUEST_PROMPT,
            },
            {
                "mode": "create_file",
                "name": "Create File",
                "description": CREATE_FILE_PROMPT,
            },
            {
                "mode": "read_file",
                "name": "Read File",
                "description": READ_FILE_PROMPT,
            },
            {
                "mode": "update_file",
                "name": "Update File",
                "description": UPDATE_FILE_PROMPT,
            },
            {
                "mode": "delete_file",
                "name": "Delete File",
                "description": DELETE_FILE_PROMPT,
            },
        ]
        tools = [
            GitLabAction(
                name=action["name"],
                description=action["description"],
                mode=action["mode"],
                api_wrapper=gitlab_api_wrapper,
            )
            for action in operations
        ]
        return cls(tools=tools)  # type: ignore[arg-type]

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return self.tools
