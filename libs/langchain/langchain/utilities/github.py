"""Util that calls GitHub."""
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, root_validator

from langchain.utils import get_from_dict_or_env


class GitHubAPIWrapper(BaseModel):
    """Wrapper for GitHub API."""

    github: Any  #: :meta private:
    github_repo_instance: Any  #: :meta private:
    github_repository: Optional[str] = None
    github_app_id: Optional[str] = None
    github_app_private_key: Optional[str] = None
    github_branch: Optional[str] = None

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

        return values

    def parse_issues(self, issues: List[dict]) -> List[dict]:
        parsed = []
        for issue in issues:
            title = issue["title"]
            number = issue["number"]
            parsed.append({"title": title, "number": number})
        return parsed

    def get_issues(self) -> str:
        issues = self.github_repo_instance.get_issues(state="open")
        parsed_issues = self.parse_issues(issues)
        parsed_issues_str = (
            "Found " + str(len(parsed_issues)) + " issues:\n" + str(parsed_issues)
        )
        return parsed_issues_str

    def get_issue(self, issue_number: int) -> Dict[str, Any]:
        issue = self.github_repo_instance.get_issue(number=issue_number)

        # If there are too many comments
        # We can't add them all to context so for now we'll just skip
        if issue.get_comments().totalCount > 10:
            return {
                "message": (
                    "There are too many comments to add them all to context. "
                    "Please visit the issue on GitHub to see them all."
                )
            }
        page = 0
        comments = []
        while True:
            comments_page = issue.get_comments().get_page(page)
            if len(comments_page) == 0:
                break
            for comment in comments_page:
                comments.append(
                    {"body": comment["body"], "user": comment["user"]["login"]}
                )
            page += 1

        return {
            "title": issue["title"],
            "body": issue["body"],
            "comments": str(comments),
        }

    def comment_on_issue(self, comment_query: str) -> str:
        # comment_query is a string which contains the issue number and the comment
        # the issue number is the first word in the string
        # the comment is the rest of the string
        issue_number = int(comment_query.split("\n\n")[0])
        comment = comment_query[len(str(issue_number)) + 2 :]

        issue = self.github_repo_instance.get_issue(number=issue_number)
        issue.create_comment(comment)
        return "Commented on issue " + str(issue_number)

    def create_file(self, file_query: str) -> str:
        # file_query is a string which contains the file path and the file contents
        # the file path is the first line in the string
        # the file contents is the rest of the string
        file_path = file_query.split("\n")[0]
        file_contents = file_query[len(file_path) + 2 :]

        self.github_repo_instance.create_file(
            path=file_path,
            message="Create " + file_path,
            content=file_contents,
            branch=self.github_branch,
        )
        return "Created file " + file_path

    def read_file(self, file_path: str) -> str:
        # file_path is a string which contains the file path
        file = self.github_repo_instance.get_contents(file_path)
        return file.decoded_content.decode("utf-8")

    def update_file(self, file_query: str) -> str:
        # file_query is a string which contains the file path and the file contents
        # the file path is the first line in the string
        # the old file contents is wrapped in OLD <<<< and >>>> OLD
        # the new file contents is wrapped in NEW <<<< and >>>> NEW

        # for example:

        # /test/test.txt
        # OLD <<<<
        # old contents
        # >>>> OLD
        # NEW <<<<
        # new contents
        # >>>> NEW

        # the old contents will be replaced with the new contents
        file_path = file_query.split("\n")[0]
        old_file_contents = file_query.split("OLD <<<<")[1].split(">>>> OLD")[0].strip()
        new_file_contents = file_query.split("NEW <<<<")[1].split(">>>> NEW")[0].strip()

        file_content = self.read_file(file_path)
        updated_file_content = file_content.replace(
            old_file_contents, new_file_contents
        )

        if file_content == updated_file_content:
            return (
                "File content was not updated because the old content was not found. "
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

    def delete_file(self, file_path: str) -> str:
        # file_path is a string which contains the file path
        file = self.github_repo_instance.get_contents(file_path)
        self.github_repo_instance.delete_file(
            path=file_path,
            message="Delete " + file_path,
            branch=self.github_branch,
            sha=file.sha,
        )
        return "Deleted file " + file_path

    def run(self, mode: str, query: str) -> str:
        if mode == "get_issues":
            return self.get_issues()
        elif mode == "get_issue":
            return json.dumps(self.get_issue(int(query)))
        elif mode == "comment_on_issue":
            return self.comment_on_issue(query)
        elif mode == "create_file":
            return self.create_file(query)
        elif mode == "read_file":
            return self.read_file(query)
        elif mode == "update_file":
            return self.update_file(query)
        elif mode == "delete_file":
            return self.delete_file(query)
        else:
            raise ValueError("Invalid mode" + mode)
