"""GitLab Toolkit."""

from typing import Dict, List, Optional

from langchain_core.tools import BaseTool
from langchain_core.tools.base import BaseToolkit

from langchain_community.tools.gitlab.prompt import (
    COMMENT_ON_ISSUE_PROMPT,
    CREATE_FILE_PROMPT,
    CREATE_PULL_REQUEST_PROMPT,
    CREATE_REPO_BRANCH,
    DELETE_FILE_PROMPT,
    GET_ISSUE_PROMPT,
    GET_ISSUES_PROMPT,
    GET_REPO_FILES_FROM_DIRECTORY,
    GET_REPO_FILES_IN_BOT_BRANCH,
    GET_REPO_FILES_IN_MAIN,
    LIST_REPO_BRANCES,
    READ_FILE_PROMPT,
    SET_ACTIVE_BRANCH,
    UPDATE_FILE_PROMPT,
)
from langchain_community.tools.gitlab.tool import GitLabAction
from langchain_community.utilities.gitlab import GitLabAPIWrapper

# only include a subset of tools by default to avoid a breaking change, where
# new tools are added to the toolkit and the user's code breaks because of
# the new tools
DEFAULT_INCLUDED_TOOLS = [
    "get_issues",
    "get_issue",
    "comment_on_issue",
    "create_pull_request",
    "create_file",
    "read_file",
    "update_file",
    "delete_file",
]


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
        cls,
        gitlab_api_wrapper: GitLabAPIWrapper,
        *,
        included_tools: Optional[List[str]] = None,
    ) -> "GitLabToolkit":
        """Create a GitLabToolkit from a GitLabAPIWrapper.

        Args:
            gitlab_api_wrapper: GitLabAPIWrapper. The GitLab API wrapper.

        Returns:
            GitLabToolkit. The GitLab toolkit.
        """

        tools_to_include = (
            included_tools if included_tools is not None else DEFAULT_INCLUDED_TOOLS
        )

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
            {
                "mode": "create_branch",
                "name": "Create a new branch",
                "description": CREATE_REPO_BRANCH,
            },
            {
                "mode": "list_branches_in_repo",
                "name": "Get the list of branches",
                "description": LIST_REPO_BRANCES,
            },
            {
                "mode": "set_active_branch",
                "name": "Change the active branch",
                "description": SET_ACTIVE_BRANCH,
            },
            {
                "mode": "list_files_in_main_branch",
                "name": "Overview of existing files in Main branch",
                "description": GET_REPO_FILES_IN_MAIN,
            },
            {
                "mode": "list_files_in_bot_branch",
                "name": "Overview of files in current working branch",
                "description": GET_REPO_FILES_IN_BOT_BRANCH,
            },
            {
                "mode": "list_files_from_directory",
                "name": "Overview of files in current working branch from a specific path",  # noqa: E501
                "description": GET_REPO_FILES_FROM_DIRECTORY,
            },
        ]
        operations_filtered = [
            operation
            for operation in operations
            if operation["mode"] in tools_to_include
        ]
        tools = [
            GitLabAction(
                name=action["name"],
                description=action["description"],
                mode=action["mode"],
                api_wrapper=gitlab_api_wrapper,
            )
            for action in operations_filtered
        ]
        return cls(tools=tools)  # type: ignore[arg-type]

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return self.tools
