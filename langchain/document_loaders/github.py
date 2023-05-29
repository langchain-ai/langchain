from abc import ABC
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

import requests
from pydantic import BaseModel, root_validator, validator

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.utils import get_from_dict_or_env


class BaseGitHubLoader(BaseLoader, BaseModel, ABC):
    """Load issues of a GitHub repository."""

    repo: str
    """Name of repository"""
    access_token: str
    """Personal access token - see https://github.com/settings/tokens?type=beta"""

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that access token exists in environment."""
        values["access_token"] = get_from_dict_or_env(
            values, "access_token", "GITHUB_PERSONAL_ACCESS_TOKEN"
        )
        return values

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.access_token}",
        }


class GitHubIssuesLoader(BaseGitHubLoader):
    include_prs: bool = True
    """If True include Pull Requests in results, otherwise ignore them."""
    milestone: Union[int, Literal["*", "none"], None] = None
    """If integer is passed, it should be a milestone's number field.
        If the string * is passed, issues with any milestone are accepted.
        If the string none is passed, issues without milestones are returned.
    """
    state: Literal["open", "closed", "all"] = "open"
    """Filter on issue state. Can be one of: open, closed, all. Default is 'open'."""
    assignee: Optional[Literal["*", "none"]] = None
    """Filter on assigned user. Pass none for no user and * for any user."""
    creator: Optional[str] = None
    """Filter on the user that created the issue."""
    mentioned: Optional[str] = None
    """Filter on a user that's mentioned in the issue."""
    labels: Optional[List[str]] = None
    """Label names to filter one. Example: bug,ui,@high."""
    sort: Literal["created", "updated", "comments"] = "created"
    """What to sort results by. Can be one of: created, updated, comments.
        Default is 'created'."""
    direction: Literal["asc", "desc"] = "desc"
    """The direction to sort the results by. Can be one of: asc, desc.
        Default is 'desc'."""
    since: Optional[str] = None
    """Only show notifications updated after the given time.
        This is a timestamp in ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ."""

    @validator("since")
    def validate_since(cls, v: Optional[str]) -> Optional[str]:
        if v:
            try:
                datetime.strptime(v, "%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                raise ValueError(
                    "Invalid value for 'since'. Expected a date string in "
                    f"YYYY-MM-DDTHH:MM:SSZ format. Received: {v}"
                )
        return v

    def load(self) -> List[Document]:
        """
        Get issues of a GitHub repository.

        Returns:
            A list of Documents with attributes:
                - page_content
                - metadata
                    - url
                    - title
                    - creator
                    - created_at
                    - last_update_time
                    - closed_time
                    - number of comments
                    - state
                    - labels
                    - assignee
                    - assignees
                    - milestone
                    - locked
                    - number
                    - is_pull_request
        """
        response = requests.get(self.url, headers=self.headers)
        response.raise_for_status()
        issues = response.json()
        documents = [self.parse_issue(issue) for issue in issues]
        if self.include_prs:
            return documents
        else:
            return [doc for doc in documents if not doc.metadata["is_pull_request"]]

    def parse_issue(self, issue: dict) -> Document:
        """Create Document objects from a list of GitHub issues."""
        metadata = {
            "url": issue["html_url"],
            "title": issue["title"],
            "creator": issue["user"]["login"],
            "created_at": issue["created_at"],
            "comments": issue["comments"],
            "state": issue["state"],
            "labels": [label["name"] for label in issue["labels"]],
            "assignee": issue["assignee"]["login"] if issue["assignee"] else None,
            "milestone": issue["milestone"]["title"] if issue["milestone"] else None,
            "locked": issue["locked"],
            "number": issue["number"],
            "is_pull_request": "pull_request" in issue,
        }
        return Document(page_content=issue["body"], metadata=metadata)

    @property
    def query_params(self) -> str:
        labels = ",".join(self.labels) if self.labels else self.labels
        query_params_dict = {
            "milestone": self.milestone,
            "state": self.state,
            "assignee": self.assignee,
            "creator": self.creator,
            "mentioned": self.mentioned,
            "labels": labels,
            "sort": self.sort,
            "direction": self.direction,
            "since": self.since,
        }
        query_params_list = [
            f"{k}={v}" for k, v in query_params_dict.items() if v is not None
        ]
        query_params = "&".join(query_params_list)
        return query_params

    @property
    def url(self) -> str:
        return f"https://api.github.com/repos/{self.repo}/issues?{self.query_params}"
