"""Util that calls gitlab."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from langchain.utils import get_from_dict_or_env

if TYPE_CHECKING:
    from gitlab.v4.objects import Issue


class GitLabAPIWrapper(BaseModel):
    """Wrapper for GitLab API."""

    gitlab: Any  #: :meta private:
    gitlab_repo_instance: Any  #: :meta private:
    gitlab_repository: Optional[str] = None
    """The name of the GitLab repository, in the form {username}/{repo-name}."""
    gitlab_personal_access_token: Optional[str] = None
    """Personal access token for the GitLab service, used for authentication."""
    gitlab_branch: Optional[str] = None
    """The specific branch in the GitLab repository where the bot will make 
        its commits. Defaults to 'main'.
    """
    gitlab_base_branch: Optional[str] = None
    """The base branch in the GitLab repository, used for comparisons. 
        Usually 'main' or 'master'. Defaults to 'main'.
    """

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        gitlab_repository = get_from_dict_or_env(
            values, "gitlab_repository", "GITLAB_REPOSITORY"
        )

        gitlab_personal_access_token = get_from_dict_or_env(
            values, "gitlab_personal_access_token", "GITLAB_PERSONAL_ACCESS_TOKEN"
        )

        gitlab_branch = get_from_dict_or_env(
            values, "gitlab_branch", "GITLAB_BRANCH", default="main"
        )
        gitlab_base_branch = get_from_dict_or_env(
            values, "gitlab_base_branch", "GITLAB_BASE_BRANCH", default="main"
        )

        try:
            import gitlab

        except ImportError:
            raise ImportError(
                "python-gitlab is not installed. "
                "Please install it with `pip install python-gitlab`"
            )

        g = gitlab.Gitlab(private_token=gitlab_personal_access_token)

        g.auth()

        values["gitlab"] = g
        values["gitlab_repo_instance"] = g.projects.get(gitlab_repository)
        values["gitlab_repository"] = gitlab_repository
        values["gitlab_personal_access_token"] = gitlab_personal_access_token
        values["gitlab_branch"] = gitlab_branch
        values["gitlab_base_branch"] = gitlab_base_branch

        return values

    def parse_issues(self, issues: List[Issue]) -> List[dict]:
        """
        Extracts title and number from each Issue and puts them in a dictionary
        Parameters:
            issues(List[Issue]): A list of gitlab Issue objects
        Returns:
            List[dict]: A dictionary of issue titles and numbers
        """
        parsed = []
        for issue in issues:
            title = issue.title
            number = issue.iid
            parsed.append({"title": title, "number": number})
        return parsed

    def get_issues(self) -> str:
        """
        Fetches all open issues from the repo

        Returns:
            str: A plaintext report containing the number of issues
            and each issue's title and number.
        """
        issues = self.gitlab_repo_instance.issues.list(state="opened")
        if len(issues) > 0:
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
            issue_number(int): The number for the gitlab issue
        Returns:
            dict: A dictionary containing the issue's title,
            body, and comments as a string
        """
        issue = self.gitlab_repo_instance.issues.get(issue_number)
        page = 0
        comments: List[dict] = []
        while len(comments) <= 10:
            comments_page = issue.notes.list(page=page)
            if len(comments_page) == 0:
                break
            for comment in comments_page:
                comment = issue.notes.get(comment.id)
                comments.append(
                    {"body": comment.body, "user": comment.author["username"]}
                )
            page += 1

        return {
            "title": issue.title,
            "body": issue.description,
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
        if self.gitlab_base_branch == self.gitlab_branch:
            return """Cannot make a pull request because 
            commits are already in the master branch"""
        else:
            try:
                title = pr_query.split("\n")[0]
                body = pr_query[len(title) + 2 :]
                pr = self.gitlab_repo_instance.mergerequests.create(
                    {
                        "source_branch": self.gitlab_branch,
                        "target_branch": self.gitlab_base_branch,
                        "title": title,
                        "description": body,
                        "labels": ["created-by-agent"],
                    }
                )
                return f"Successfully created PR number {pr.iid}"
            except Exception as e:
                return "Unable to make pull request due to error:\n" + str(e)

    def comment_on_issue(self, comment_query: str) -> str:
        """
        Adds a comment to a gitlab issue
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
            issue = self.gitlab_repo_instance.issues.get(issue_number)
            issue.notes.create({"body": comment})
            return "Commented on issue " + str(issue_number)
        except Exception as e:
            return "Unable to make comment due to error:\n" + str(e)

    def create_file(self, file_query: str) -> str:
        """
        Creates a new file on the gitlab repo
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
            self.gitlab_repo_instance.files.get(file_path, self.gitlab_branch)
            return f"File already exists at {file_path}. Use update_file instead"
        except Exception:
            data = {
                "branch": self.gitlab_branch,
                "commit_message": "Create " + file_path,
                "file_path": file_path,
                "content": file_contents,
            }

            self.gitlab_repo_instance.files.create(data)

            return "Created file " + file_path

    def read_file(self, file_path: str) -> str:
        """
        Reads a file from the gitlab repo
        Parameters:
            file_path(str): the file path
        Returns:
            str: The file decoded as a string
        """
        file = self.gitlab_repo_instance.files.get(file_path, self.gitlab_branch)
        return file.decode().decode("utf-8")

    def update_file(self, file_query: str) -> str:
        """
        Updates a file with new content.
        Parameters:
            file_query(str): Contains the file path and the file contents.
                The old file contents is wrapped in OLD <<<< and >>>> OLD
                The new file contents is wrapped in NEW <<<< and >>>> NEW
                For example:
                test/hello.txt
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

            commit = {
                "branch": self.gitlab_branch,
                "commit_message": "Create " + file_path,
                "actions": [
                    {
                        "action": "update",
                        "file_path": file_path,
                        "content": updated_file_content,
                    }
                ],
            }

            self.gitlab_repo_instance.commits.create(commit)
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
            self.gitlab_repo_instance.files.delete(
                file_path, self.gitlab_branch, "Delete " + file_path
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
