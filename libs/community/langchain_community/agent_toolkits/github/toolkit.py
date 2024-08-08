"""GitHub Toolkit."""

from typing import Dict, List

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseToolkit

from langchain_community.tools import BaseTool
from langchain_community.tools.github.prompt import (
    COMMENT_ON_ISSUE_PROMPT,
    CREATE_BRANCH_PROMPT,
    CREATE_FILE_PROMPT,
    CREATE_PULL_REQUEST_PROMPT,
    CREATE_REVIEW_REQUEST_PROMPT,
    DELETE_FILE_PROMPT,
    GET_FILES_FROM_DIRECTORY_PROMPT,
    GET_ISSUE_PROMPT,
    GET_ISSUES_PROMPT,
    GET_PR_PROMPT,
    LIST_BRANCHES_IN_REPO_PROMPT,
    LIST_PRS_PROMPT,
    LIST_PULL_REQUEST_FILES,
    OVERVIEW_EXISTING_FILES_BOT_BRANCH,
    OVERVIEW_EXISTING_FILES_IN_MAIN,
    READ_FILE_PROMPT,
    SEARCH_CODE_PROMPT,
    SEARCH_ISSUES_AND_PRS_PROMPT,
    SET_ACTIVE_BRANCH_PROMPT,
    UPDATE_FILE_PROMPT,
)
from langchain_community.tools.github.tool import GitHubAction
from langchain_community.utilities.github import GitHubAPIWrapper


class NoInput(BaseModel):
    """Schema for operations that do not require any input."""

    no_input: str = Field("", description="No input required, e.g. `` (empty string).")


class GetIssue(BaseModel):
    """Schema for operations that require an issue number as input."""

    issue_number: int = Field(0, description="Issue number as an integer, e.g. `42`")


class CommentOnIssue(BaseModel):
    """Schema for operations that require a comment as input."""

    input: str = Field(..., description="Follow the required formatting.")


class GetPR(BaseModel):
    """Schema for operations that require a PR number as input."""

    pr_number: int = Field(0, description="The PR number as an integer, e.g. `12`")


class CreatePR(BaseModel):
    """Schema for operations that require a PR title and body as input."""

    formatted_pr: str = Field(..., description="Follow the required formatting.")


class CreateFile(BaseModel):
    """Schema for operations that require a file path and content as input."""

    formatted_file: str = Field(..., description="Follow the required formatting.")


class ReadFile(BaseModel):
    """Schema for operations that require a file path as input."""

    formatted_filepath: str = Field(
        ...,
        description=(
            "The full file path of the file you would like to read where the "
            "path must NOT start with a slash, e.g. `some_dir/my_file.py`."
        ),
    )


class UpdateFile(BaseModel):
    """Schema for operations that require a file path and content as input."""

    formatted_file_update: str = Field(
        ..., description="Strictly follow the provided rules."
    )


class DeleteFile(BaseModel):
    """Schema for operations that require a file path as input."""

    formatted_filepath: str = Field(
        ...,
        description=(
            "The full file path of the file you would like to delete"
            " where the path must NOT start with a slash, e.g."
            " `some_dir/my_file.py`. Only input a string,"
            " not the param name."
        ),
    )


class DirectoryPath(BaseModel):
    """Schema for operations that require a directory path as input."""

    input: str = Field(
        "",
        description=(
            "The path of the directory, e.g. `some_dir/inner_dir`."
            " Only input a string, do not include the parameter name."
        ),
    )


class BranchName(BaseModel):
    """Schema for operations that require a branch name as input."""

    branch_name: str = Field(
        ..., description="The name of the branch, e.g. `my_branch`."
    )


class SearchCode(BaseModel):
    """Schema for operations that require a search query as input."""

    search_query: str = Field(
        ...,
        description=(
            "A keyword-focused natural language search"
            "query for code, e.g. `MyFunctionName()`."
        ),
    )


class CreateReviewRequest(BaseModel):
    """Schema for operations that require a username as input."""

    username: str = Field(
        ...,
        description="GitHub username of the user being requested, e.g. `my_username`.",
    )


class SearchIssuesAndPRs(BaseModel):
    """Schema for operations that require a search query as input."""

    search_query: str = Field(
        ...,
        description="Natural language search query, e.g. `My issue title or topic`.",
    )


class GitHubToolkit(BaseToolkit):
    """GitHub Toolkit.

    *Security Note*: This toolkit contains tools that can read and modify
        the state of a service; e.g., by creating, deleting, or updating,
        reading underlying data.

        For example, this toolkit can be used to create issues, pull requests,
        and comments on GitHub.

        See [Security](https://python.langchain.com/docs/security) for more information.

    Setup:
        See detailed installation instructions here:
        https://python.langchain.com/v0.2/docs/integrations/tools/github/#installation

        You will need to install ``pygithub`` and set the following environment
        variables:

        .. code-block:: bash

            pip install -U pygithub
            export GITHUB_APP_ID="your-app-id"
            export GITHUB_APP_PRIVATE_KEY="path-to-private-key"
            export GITHUB_REPOSITORY="your-github-repository"

    Instantiate:
        .. code-block:: python

            from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
            from langchain_community.utilities.github import GitHubAPIWrapper

            github = GitHubAPIWrapper()
            toolkit = GitHubToolkit.from_github_api_wrapper(github)

    Tools:
        .. code-block:: python

            tools = toolkit.get_tools()
            for tool in tools:
                print(tool.name)

        .. code-block:: none

            Get Issues
            Get Issue
            Comment on Issue
            List open pull requests (PRs)
            Get Pull Request
            Overview of files included in PR
            Create Pull Request
            List Pull Requests' Files
            Create File
            Read File
            Update File
            Delete File
            Overview of existing files in Main branch
            Overview of files in current working branch
            List branches in this repository
            Set active branch
            Create a new branch
            Get files from a directory
            Search issues and pull requests
            Search code
            Create review request

    Use within an agent:
        .. code-block:: python

            from langchain_openai import ChatOpenAI
            from langgraph.prebuilt import create_react_agent

            # Select example tool
            tools = [tool for tool in toolkit.get_tools() if tool.name == "Get Issue"]
            assert len(tools) == 1
            tools[0].name = "get_issue"

            llm = ChatOpenAI(model="gpt-4o-mini")
            agent_executor = create_react_agent(llm, tools)

            example_query = "What is the title of issue 24888?"

            events = agent_executor.stream(
                {"messages": [("user", example_query)]},
                stream_mode="values",
            )
            for event in events:
                event["messages"][-1].pretty_print()

        .. code-block:: none

             ================================[1m Human Message [0m=================================

            What is the title of issue 24888?
            ==================================[1m Ai Message [0m==================================
            Tool Calls:
            get_issue (call_iSYJVaM7uchfNHOMJoVPQsOi)
            Call ID: call_iSYJVaM7uchfNHOMJoVPQsOi
            Args:
                issue_number: 24888
            =================================[1m Tool Message [0m=================================
            Name: get_issue

            {"number": 24888, "title": "Standardize KV-Store Docs", "body": "..."
            ==================================[1m Ai Message [0m==================================

            The title of issue 24888 is "Standardize KV-Store Docs".

    Parameters:
        tools: List[BaseTool]. The tools in the toolkit. Default is an empty list.
    """  # noqa: E501

    tools: List[BaseTool] = []

    @classmethod
    def from_github_api_wrapper(
        cls, github_api_wrapper: GitHubAPIWrapper
    ) -> "GitHubToolkit":
        """Create a GitHubToolkit from a GitHubAPIWrapper.

        Args:
            github_api_wrapper: GitHubAPIWrapper. The GitHub API wrapper.

        Returns:
            GitHubToolkit. The GitHub toolkit.
        """
        operations: List[Dict] = [
            {
                "mode": "get_issues",
                "name": "Get Issues",
                "description": GET_ISSUES_PROMPT,
                "args_schema": NoInput,
            },
            {
                "mode": "get_issue",
                "name": "Get Issue",
                "description": GET_ISSUE_PROMPT,
                "args_schema": GetIssue,
            },
            {
                "mode": "comment_on_issue",
                "name": "Comment on Issue",
                "description": COMMENT_ON_ISSUE_PROMPT,
                "args_schema": CommentOnIssue,
            },
            {
                "mode": "list_open_pull_requests",
                "name": "List open pull requests (PRs)",
                "description": LIST_PRS_PROMPT,
                "args_schema": NoInput,
            },
            {
                "mode": "get_pull_request",
                "name": "Get Pull Request",
                "description": GET_PR_PROMPT,
                "args_schema": GetPR,
            },
            {
                "mode": "list_pull_request_files",
                "name": "Overview of files included in PR",
                "description": LIST_PULL_REQUEST_FILES,
                "args_schema": GetPR,
            },
            {
                "mode": "create_pull_request",
                "name": "Create Pull Request",
                "description": CREATE_PULL_REQUEST_PROMPT,
                "args_schema": CreatePR,
            },
            {
                "mode": "list_pull_request_files",
                "name": "List Pull Requests' Files",
                "description": LIST_PULL_REQUEST_FILES,
                "args_schema": GetPR,
            },
            {
                "mode": "create_file",
                "name": "Create File",
                "description": CREATE_FILE_PROMPT,
                "args_schema": CreateFile,
            },
            {
                "mode": "read_file",
                "name": "Read File",
                "description": READ_FILE_PROMPT,
                "args_schema": ReadFile,
            },
            {
                "mode": "update_file",
                "name": "Update File",
                "description": UPDATE_FILE_PROMPT,
                "args_schema": UpdateFile,
            },
            {
                "mode": "delete_file",
                "name": "Delete File",
                "description": DELETE_FILE_PROMPT,
                "args_schema": DeleteFile,
            },
            {
                "mode": "list_files_in_main_branch",
                "name": "Overview of existing files in Main branch",
                "description": OVERVIEW_EXISTING_FILES_IN_MAIN,
                "args_schema": NoInput,
            },
            {
                "mode": "list_files_in_bot_branch",
                "name": "Overview of files in current working branch",
                "description": OVERVIEW_EXISTING_FILES_BOT_BRANCH,
                "args_schema": NoInput,
            },
            {
                "mode": "list_branches_in_repo",
                "name": "List branches in this repository",
                "description": LIST_BRANCHES_IN_REPO_PROMPT,
                "args_schema": NoInput,
            },
            {
                "mode": "set_active_branch",
                "name": "Set active branch",
                "description": SET_ACTIVE_BRANCH_PROMPT,
                "args_schema": BranchName,
            },
            {
                "mode": "create_branch",
                "name": "Create a new branch",
                "description": CREATE_BRANCH_PROMPT,
                "args_schema": BranchName,
            },
            {
                "mode": "get_files_from_directory",
                "name": "Get files from a directory",
                "description": GET_FILES_FROM_DIRECTORY_PROMPT,
                "args_schema": DirectoryPath,
            },
            {
                "mode": "search_issues_and_prs",
                "name": "Search issues and pull requests",
                "description": SEARCH_ISSUES_AND_PRS_PROMPT,
                "args_schema": SearchIssuesAndPRs,
            },
            {
                "mode": "search_code",
                "name": "Search code",
                "description": SEARCH_CODE_PROMPT,
                "args_schema": SearchCode,
            },
            {
                "mode": "create_review_request",
                "name": "Create review request",
                "description": CREATE_REVIEW_REQUEST_PROMPT,
                "args_schema": CreateReviewRequest,
            },
        ]
        tools = [
            GitHubAction(
                name=action["name"],
                description=action["description"],
                mode=action["mode"],
                api_wrapper=github_api_wrapper,
                args_schema=action.get("args_schema", None),
            )
            for action in operations
        ]
        return cls(tools=tools)  # type: ignore[arg-type]

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return self.tools
