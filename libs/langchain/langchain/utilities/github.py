"""Util that calls GitHub."""
from __future__ import annotations

import json
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import requests
import tiktoken
from langchain.utils import get_from_dict_or_env
from pydantic import BaseModel, Extra, root_validator

if TYPE_CHECKING:
    from github.Issue import Issue
    from github.PullRequest import PullRequest


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

        try:
            with open(github_app_private_key, "r") as f:
                private_key = f.read()
        except FileNotFoundError as e:
            if type(github_app_private_key) == str:
                private_key == github_app_private_key
            else:
                raise FileNotFoundError(f"Github App private key cannot be found in filesystem, and is not a string. Found: {github_app_private_key}")

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
    
    def parse_pull_requests(self, pull_requests: List[PullRequest]) -> List[dict]:
        """
        Extracts title and number from each Issue and puts them in a dictionary
        Parameters:
            issues(List[Issue]): A list of Github Issue objects
        Returns:
            List[dict]: A dictionary of issue titles and numbers
        """
        parsed = []
        for pr in pull_requests:
            title = pr.title
            number = pr.number
            commits = pr.commits
            parsed.append({"title": title, "number": number, "commits": str(commits)})
        
        print("❤️❤️❤️❤️❤️❤️ PARSED PRS:")
        print(parsed)
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
    
    def list_open_pull_requests(self) -> str:
        """
        Fetches all open PRs from the repo

        Returns:
            str: A plaintext report containing the number of PRs
            and each PR's title and number.
        """
        # issues = self.github_repo_instance.get_issues(state="open")
        print("⭐️⭐️⭐️⭐️⭐️⭐️ IN LIST OPEN PRS")
        pull_requests = self.github_repo_instance.get_pulls(state="open")
        print(f"⭐️PRS: {pull_requests}")
        if pull_requests.totalCount > 0:
            parsed_prs = self.parse_pull_requests(pull_requests)
            parsed_prs_str = (
                "Found " + str(len(parsed_prs)) + " pull requests:\n" + str(parsed_prs)
            )
            return parsed_prs_str
        else:
            return "No open pull requests available"

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
    
    def list_pull_request_files(self, pr_number: int) -> Dict[str, Any]:
        """Fetches the full text of all files in a PR. Truncates after first 3k tokens. 
        # TODO: Enhancement to summarize files with ctags if they're getting long.

        Args:
            pr_number(int): The number of the pull request on Github

        Returns:
            dict: A dictionary containing the issue's title,
            body, and comments as a string
        """
        MAX_TOKENS_FOR_FILES = 3_000
        pr_files = []
        pr = self.github_repo_instance.get_pull(number=int(pr_number))
        total_tokens = 0
        page=0
        # print(f"Top of get files from PR")
        
        while True: # (total_tokens + tiktoken()) < MAX_TOKENS_FOR_FILES:
            files_page = pr.get_files().get_page(page)
            if len(files_page) == 0:
                break
            for file in files_page:
                try:
                    file_metadata_response = requests.get(file.contents_url)
                    if file_metadata_response.status_code == 200:
                        download_url = json.loads(file_metadata_response.text)['download_url']
                    else:
                        print(f"❌❌ Failed to download file: {file.contents_url}, skipping")
                        continue
                    
                    file_content_response = requests.get(download_url)
                    if file_content_response.status_code == 200:
                        # Save the content as a UTF-8 string
                        file_content = file_content_response.text
                        # print("File content", file_content)
                    else:
                        print(f"failed downloading file content (Error {file_content_response.status_code}). Skipping")
                        continue
                    
                    file_tokens = len(tiktoken.get_encoding("cl100k_base").encode(file_content + file.filename + "file_name file_contents"))
                    # print(f"Getting file contents from Github: {file_content}")
                    if (total_tokens + file_tokens) < MAX_TOKENS_FOR_FILES:
                        pr_files.append({"filename": file.filename, "contents": file_content,"additions": file.additions,"deletions": file.deletions})
                        total_tokens += file_tokens
                except Exception as e:
                    print(f"Error when reading files from a PR on github. {e}")
                    # pr_files.append({"file_name": f"Error reading file: {e}" ,"file_contents": "None"})
            page += 1
        return pr_files
    
    def get_pull_request(self, pr_number: int) -> Dict[str, Any]:
        """
        Fetches a specific pull request and its first 10 comments
        Parameters:
            pr_number(int): The number for the github pull
        Returns:
            dict: A dictionary containing the pull's title,
            body, and comments as a string
        """
        pull = self.github_repo_instance.get_pull(number=pr_number)
        page = 0
        comments: List[dict] = []
        while len(comments) <= 10:
            # For normal conversation comments use get_issue_comments (even on PRs)
            comments_page = pull.get_issue_comments().get_page(page)
            if len(comments_page) == 0:
                break
            for comment in comments_page:
                comments.append({"body": comment.body, "user": comment.user.login})
            page += 1
        
        page = 0
        commits: List[dict] = []
        while len(commits) <= 10:
            comments_page = pull.get_commits().get_page(page)
            if len(comments_page) == 0:
                break
            for commit in comments_page:
                commits.append({"message": commit.commit.message, "sha": commit.commit.sha})
            page += 1

        return {
            "title": pull.title,
            "number": pr_number,
            "body": pull.body,
            "comments": str(comments),
            "commits": str(commits),
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
            try: 
                self.github_repo_instance.get_contents(file_path)
            except Exception as e:
                return f"File already exists at {file_path} (on branch {self.github_branch}). Use update_file instead."
            self.github_repo_instance.create_file(
                path=file_path,
                message="Create " + file_path,
                content=file_contents,
                branch=self.github_branch,
            )
            return "Created file " + file_path
        except Exception as e:
            return "Unable to make file due to error:\n" + str(e)

    def read_file(self, file_path: str) -> str:
        """
        Read a file from this agent's branch, defined by self.github_branch, which supports PR branches.
        Parameters:
            file_path(str): the file path
        Returns:
            str: The file decoded as a string, or an error message if not found
        """
        try:
            file = self.github_repo_instance.get_contents(file_path, ref=self.github_branch)
            return file.decoded_content.decode("utf-8")
        except Exception as e:
            return f"File not found at {file_path} in branch {self.github_branch}. Error: {str(e)}"


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
                sha=self.github_repo_instance.get_contents(file_path, ref=self.github_branch).sha, 
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
            self.github_repo_instance.delete_file(
                path=file_path,
                message="Delete " + file_path,
                branch=self.github_branch,
                sha=self.github_repo_instance.get_contents(file_path, ref=self.github_branch).sha,
            )
            return "Deleted file " + file_path
        except Exception as e:
            return "Unable to delete file due to error:\n" + str(e)
    
    def search_issues_and_prs(self, query: str) -> str:
        """
        Searches issues and pull requests in the repository.
        
        Parameters:
            query(str): The search query
        
        Returns:
            str: A string containing the first 5 issues and pull requests
        """
        search_result = self.github.search_issues(query, repo=self.github_repository)
        results = []
        for issue in search_result[:5]:
            results.append(f"Title: {issue.title}, Number: {issue.number}, State: {issue.state}")
        
        return "\n".join(results)

    def search_code(self, query: str) -> str:
        """
        Searches code in the repository.
        
        Parameters:
            query(str): The search query
        
        Returns:
            str: A string containing the first 5 code results
        """
        search_result = self.github.search_code(query, repo=self.github_repository)
        results = []
        for code in search_result[:5]:
            # TODO: return the full code, not just the URL to it
            results.append(f"Path: {code.path}, URL: {code.html_url}")
        
        return "\n".join(results)

    def create_review_request(self, pull_request_number: int, reviewer_username: str) -> str:
        """
        Creates a review request on an open pull request.
        
        Parameters:
            pull_request_number(int): The number of the pull request
            reviewer_username(str): The username of the person who is being requested
        
        Returns:
            str: A message confirming the creation of the review request
        """
        pull_request = self.github_repo_instance.get_pull(number=pull_request_number)
        pull_request.create_review_request(reviewers=[reviewer_username])
        
        return f"Review request created for user {reviewer_username} on PR #{pull_request_number}"

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
        elif mode == "list_open_pull_requests":
            return self.list_open_pull_requests()
        elif mode == "get_pull_request":
            return self.get_pull_request(int(query))
        elif mode == "list_pull_request_files":
            return self.list_pull_request_files(int(query))
        else:
            raise ValueError("Invalid mode" + mode)
