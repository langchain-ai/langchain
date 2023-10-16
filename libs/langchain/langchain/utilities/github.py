"""Util that calls GitHub."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from langchain.utils import get_from_dict_or_env

if TYPE_CHECKING:
    from github.Issue import Issue


class GitHubAPIWrapper(BaseModel):
    """Wrapper for GitHub API."""

    github: Any  #: :meta private:
    github_repo_instance: Any  #: :meta private:
    github_repository: Optional[str] = None
    github_app_id: Optional[str] = None
    github_app_private_key: Optional[str] = None
    github_branch: Optional[str] = None
    github_base_branch: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        github_repository = get_from_dict_or_env(
            values, "github_repository", "GITHUB_REPOSITORY"
        )

        github_app_id = get_from_dict_or_env(values, "github_app_id", "GITHUB_APP_ID")

        github_app_private_key = get_from_dict_or_env(
            values, "github_app_private_key", "GITHUB_APP_PRIVATE_KEY"
        )

        github_branch = get_from_dict_or_env(
            values, "github_branch", "GITHUB_BRANCH", default="master"
        )
        github_base_branch = get_from_dict_or_env(
            values, "github_base_branch", "GITHUB_BASE_BRANCH", default="master"
        )

        try:
            from github import Auth, GithubIntegration

        except ImportError:
            raise ImportError(
                "PyGithub is not installed. "
                "Please install it with `pip install PyGithub`"
            )

        with open(github_app_private_key, "r") as f:
            private_key = f.read()

        auth = Auth.AppAuth(
            github_app_id,
            private_key,
        )
        gi = GithubIntegration(auth=auth)
        installation = gi.get_installations()[0]

        # create a GitHub instance:
        g = installation.get_github_for_installation()

        values["github"] = g
        values["github_repo_instance"] = g.get_repo(github_repository)
        values["github_repository"] = github_repository
        values["github_app_id"] = github_app_id
        values["github_app_private_key"] = github_app_private_key
        values["github_branch"] = github_branch
        values["github_base_branch"] = github_base_branch

        return values

    def parse_issues(self, issues: List[Issue]) -> List[dict]:
        """
        Extracts title and number from each Issue and puts them in a dictionary
        Parameters:
            issues(List[Issue]): A list of Github Issue objects
        Returns:
            List[dict]: A dictionary of issue titles and numbers
        """
        parsed = []
        for issue in issues:
            title = issue.title
            number = issue.number
            parsed.append({"title": title, "number": number})
        return parsed

    def get_issues(self) -> str:
        """
        Fetches all open issues from the repo

        Returns:
            str: A plaintext report containing the number of issues
            and each issue's title and number.
        """
        issues = self.github_repo_instance.get_issues(state="open")
        if issues.totalCount > 0:
            parsed_issues = self.parse_issues(issues)
            parsed_issues_str = (
                "Found " + str(len(parsed_issues)) + " issues:\n" + str(parsed_issues)
            )
            return parsed_issues_str
        else:
            return "No open issues available"

    def get_issue(self, issue_number: int) -> Dict[str, Any]:
        """
        Fetches a specific issue and its first 10 comments
        Parameters:
            issue_number(int): The number for the github issue
        Returns:
            dict: A doctionary containing the issue's title,
            body, and comments as a string
        """
        issue = self.github_repo_instance.get_issue(number=issue_number)
        page = 0
        comments: List[dict] = []
        while len(comments) <= 10:
            comments_page = issue.get_comments().get_page(page)
            if len(comments_page) == 0:
                break
            for comment in comments_page:
                comments.append({"body": comment.body, "user": comment.user.login})
            page += 1

        return {
            "title": issue.title,
            "body": issue.body,
            "comments": str(comments),
        }

    def create_pull_request(self, pr_query: str) -> str:
        """
        Makes a pull request from the bot's branch to the base branch
        Parameters:
            pr_query(str): a string which contains the PR title
            and the PR body. The title is the first line
            in the string, and the body are the rest of the string.
            For example, "Updated README\nmade changes to add info"
        Returns:
            str: A success or failure message
        """
        if self.github_base_branch == self.github_branch:
            return """Cannot make a pull request because 
            commits are already in the master branch"""
        else:
            try:
                title = pr_query.split("\n")[0]
                body = pr_query[len(title) + 2 :]
                pr = self.github_repo_instance.create_pull(
                    title=title,
                    body=body,
                    head=self.github_branch,
                    base=self.github_base_branch,
                )
                return f"Successfully created PR number {pr.number}"
            except Exception as e:
                return "Unable to make pull request due to error:\n" + str(e)

    def comment_on_issue(self, comment_query: str) -> str:
        """
        Adds a comment to a github issue
        Parameters:
            comment_query(str): a string which contains the issue number,
            two newlines, and the comment.
            for example: "1\n\nWorking on it now"
            adds the comment "working on it now" to issue 1
        Returns:
            str: A success or failure message
        """
        issue_number = int(comment_query.split("\n\n")[0])
        comment = comment_query[len(str(issue_number)) + 2 :]
        try:
            issue = self.github_repo_instance.get_issue(number=issue_number)
            issue.create_comment(comment)
            return "Commented on issue " + str(issue_number)
        except Exception as e:
            return "Unable to make comment due to error:\n" + str(e)

    def create_file(self, file_query: str) -> str:
        """
        Creates a new file on the Github repo
        Parameters:
            file_query(str): a string which contains the file path
            and the file contents. The file path is the first line
            in the string, and the contents are the rest of the string.
            For example, "hello_world.md\n# Hello World!"
        Returns:
            str: A success or failure message
        """
        file_path = file_query.split("\n")[0]
        file_contents = file_query[len(file_path) + 2 :]
        try:
            exists = self.github_repo_instance.get_contents(file_path)
            if exists is None:
                self.github_repo_instance.create_file(
                    path=file_path,
                    message="Create " + file_path,
                    content=file_contents,
                    branch=self.github_branch,
                )
                return "Created file " + file_path
            else:
                return f"File already exists at {file_path}. Use update_file instead"
        except Exception as e:
            return "Unable to make file due to error:\n" + str(e)

    def read_file(self, file_path: str) -> str:
        """
        Reads a file from the github repo
        Parameters:
            file_path(str): the file path
        Returns:
            str: The file decoded as a string
        """
        file = self.github_repo_instance.get_contents(file_path)
        return file.decoded_content.decode("utf-8")

    def update_file(self, file_query: str) -> str:
        """
        Updates a file with new content.
        Parameters:
            file_query(str): Contains the file path and the file contents.
                The old file contents is wrapped in OLD <<<< and >>>> OLD
                The new file contents is wrapped in NEW <<<< and >>>> NEW
                For example:
                /test/hello.txt
                OLD <<<<
                Hello Earth!
                >>>> OLD
                NEW <<<<
                Hello Mars!
                >>>> NEW
        Returns:
            A success or failure message
        """
        try:
            file_path = file_query.split("\n")[0]
            old_file_contents = (
                file_query.split("OLD <<<<")[1].split(">>>> OLD")[0].strip()
            )
            new_file_contents = (
                file_query.split("NEW <<<<")[1].split(">>>> NEW")[0].strip()
            )

            file_content = self.read_file(file_path)
            updated_file_content = file_content.replace(
                old_file_contents, new_file_contents
            )

            if file_content == updated_file_content:
                return (
                    "File content was not updated because old content was not found."
                    "It may be helpful to use the read_file action to get "
                    "the current file contents."
                )

            self.github_repo_instance.update_file(
                path=file_path,
                message="Update " + file_path,
                content=updated_file_content,
                branch=self.github_branch,
                sha=self.github_repo_instance.get_contents(file_path).sha,
            )
            return "Updated file " + file_path
        except Exception as e:
            return "Unable to update file due to error:\n" + str(e)

    def delete_file(self, file_path: str) -> str:
        """
        Deletes a file from the repo
        Parameters:
            file_path(str): Where the file is
        Returns:
            str: Success or failure message
        """
        try:
            file = self.github_repo_instance.get_contents(file_path)
            self.github_repo_instance.delete_file(
                path=file_path,
                message="Delete " + file_path,
                branch=self.github_branch,
                sha=file.sha,
            )
            return "Deleted file " + file_path
        except Exception as e:
            return "Unable to delete file due to error:\n" + str(e)

    def run(self, mode: str, query: str) -> str:
        if mode == "get_issues":
            return self.get_issues()
        elif mode == "get_issue":
            return json.dumps(self.get_issue(int(query)))
        elif mode == "comment_on_issue":
            return self.comment_on_issue(query)
        elif mode == "create_file":
            return self.create_file(query)
        elif mode == "create_pull_request":
            return self.create_pull_request(query)
        elif mode == "read_file":
            return self.read_file(query)
        elif mode == "update_file":
            return self.update_file(query)
        elif mode == "delete_file":
            return self.delete_file(query)
        else:
            raise ValueError("Invalid mode" + mode)
