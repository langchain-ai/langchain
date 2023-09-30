"""GitHub Toolkit."""
from typing import Dict, List

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.tools import BaseTool
from langchain.tools.github.prompt import (COMMENT_ON_ISSUE_PROMPT,
                                           CREATE_BRANCH_PROMPT,
                                           CREATE_FILE_PROMPT,
                                           CREATE_PULL_REQUEST_PROMPT, CREATE_REVIEW_REQUEST_PROMPT,
                                           DELETE_FILE_PROMPT,
                                           GET_FILES_FROM_DIRECTORY_PROMPT,
                                           GET_ISSUE_PROMPT, GET_ISSUES_PROMPT,
                                           GET_PR_PROMPT,
                                           LIST_BRANCHES_IN_REPO_PROMPT,
                                           LIST_PRS_PROMPT,
                                           LIST_PULL_REQUEST_FILES,
                                           OVERVIEW_EXISTING_FILES_BOT_BRANCH,
                                           OVERVIEW_EXISTING_FILES_IN_MAIN,
                                           READ_FILE_PROMPT, SEARCH_CODE_PROMPT, SEARCH_ISSUES_AND_PRS_PROMPT,
                                           SET_ACTIVE_BRANCH_PROMPT,
                                           UPDATE_FILE_PROMPT)
from langchain.tools.github.tool import GitHubAction
from langchain.utilities.github import GitHubAPIWrapper


class GitHubToolkit(BaseToolkit):
    """GitHub Toolkit."""

    tools: List[BaseTool] = []

    @classmethod
    def from_github_api_wrapper(
        cls, github_api_wrapper: GitHubAPIWrapper
    ) -> "GitHubToolkit":
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
                "mode": "list_open_pull_requests",
                "name": "List open pull requests (PRs)",
                "description": LIST_PRS_PROMPT,
            },
            {
                "mode": "get_pull_request",
                "name": "Get Pull Request (fetch by PR number)",
                "description": GET_PR_PROMPT,
            },
            {
                "mode": "list_pull_request_files",
                "name": "Get an overview of the files in any Pull Request (fetch by PR number)",
                "description": LIST_PULL_REQUEST_FILES,
            },
            {
                "mode": "create_pull_request",
                "name": "Create Pull Request",
                "description": CREATE_PULL_REQUEST_PROMPT,
            },
            {
                "mode": "list_pull_request_files",
                "name": "List Pull Requests' Files",
                "description": LIST_PULL_REQUEST_FILES,
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
                "mode": "list_files_in_main_branch",
                "name": "Overview of existing files in Main branch (no input)",
                "description": OVERVIEW_EXISTING_FILES_IN_MAIN,
            },
            {
                "mode": "list_files_in_bot_branch",
                "name": "Overview of files in your current working branch",
                "description": OVERVIEW_EXISTING_FILES_BOT_BRANCH,
            },
            {
                "mode": "list_branches_in_repo",
                "name": "List branches in this repository",
                "description": LIST_BRANCHES_IN_REPO_PROMPT,
            },
            {
                "mode": "set_active_branch",
                "name": "Set active branch",
                "description": SET_ACTIVE_BRANCH_PROMPT,
            },
            {
                "mode": "create_branch",
                "name": "Create a new branch",
                "description": CREATE_BRANCH_PROMPT,
            },
            {
                "mode": "get_files_from_directory",
                "name": "Get files from a directory",
                "description": GET_FILES_FROM_DIRECTORY_PROMPT,
            },
            {
                "mode": "search_issues_and_prs",
                "name": "Search issues and pull requests",
                "description": SEARCH_ISSUES_AND_PRS_PROMPT,
            },
            {
                "mode": "search_code",
                "name": "Search code",
                "description": SEARCH_CODE_PROMPT,
            },
            {
                "mode": "create_review_request",
                "name": "Create review request",
                "description": CREATE_REVIEW_REQUEST_PROMPT,
            },
        ]
        tools = [
            GitHubAction(
                name=action["name"],
                description=action["description"],
                mode=action["mode"],
                api_wrapper=github_api_wrapper,
            )
            for action in operations
        ]
        return cls(tools=tools)

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return self.tools
