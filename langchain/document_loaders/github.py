from datetime import datetime
from typing import Any, Dict, List

import requests
from pydantic import BaseModel, root_validator

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.utils import get_from_dict_or_env


class GitHubLoader(BaseLoader, BaseModel):
    """Load issues of a GitHub repository."""

    repo: str = ""
    """Name of repository"""
    access_token: str = ""
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

    def load(self, include_prs: bool = True, **kwargs: Any) -> List[Document]:
        """
        Get issues of a GitHub repository.

        Args:
            include_prs: Should pull requests als be returned, or only issues?
            milestone:
                If integer is passed, it should be a milestone's number field.
                If the string * is passed, issues with any milestone are accepted.
                If the string none is passed, issues without milestones are returned.
            state: Can be one of: open, closed, all. Default is 'open'.
            assignee: Assigned user. Pass none for no user and * for any user.
            creator: The user that created the issue.
            mentioned: A user that's mentioned in the issue.
            labels: A list of comma separated label names. Example: bug,ui,@high.
            sort: What to sort results by. Can be one of: created, updated, comments.
                Default is 'created'.
            direction: The direction to sort the results by. Can be one of: asc, desc.
                Default is 'desc'.
            since: Only show notifications updated after the given time.
                This is a timestamp in ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ.

        Returns:
            A list of Documents with attributes:
                - page_content
                - metadata
                    - url
                    - title
                    - creator
                    - creation_time
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
        url = self.build_url(**kwargs)
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        issues = response.json()
        documents = [self.parse_issue(issue) for issue in issues]
        if include_prs:
            return documents
        else:
            return [doc for doc in documents if not doc.metadata["is_pull_request"]]

    def parse_issue(self, issue: dict) -> Document:
        """Create Document objects from a list of GitHub issues."""
        metadata = {
            "url": issue["html_url"],
            "title": issue["title"],
            "creator": issue["user"]["login"],
            "creation_time": issue["created_at"],
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

    def build_url(self, **kwargs: Any) -> str:
        valid_kwargs = {
            "milestone": {"*", "none"},  # todo: add milestone number
            "state": {"open", "closed", "all"},
            "sort": {"created", "updated", "comments"},
            "direction": {"asc", "desc"},
        }

        for key, value in kwargs.items():
            # we validate these below or not at all
            if key in ["labels", "since", "assignee", "creator", "mentioned"]:
                continue

            if key not in valid_kwargs:
                raise ValueError(f"Invalid keyword argument: {key}")
            if value not in valid_kwargs[key]:
                raise ValueError(
                    f"Invalid value for {key}: {value}. "
                    f"Expected one of {valid_kwargs[key]}"
                )

        if "labels" in kwargs and not isinstance(kwargs["labels"], list):
            raise ValueError("Invalid value for labels: Expected a list of strings")

        if "since" in kwargs:
            try:
                datetime.strptime(kwargs["since"], "%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                raise ValueError(
                    "Invalid value for since: Expected a date string in "
                    "YYYY-MM-DDTHH:MM:SSZ format"
                )

        url = f"https://api.github.com/repos/{self.repo}/issues"
        query_params_list = []
        for k, v in kwargs.items():
            if k == "labels" and isinstance(v, list):
                # labels values should be a comma-separated list of values
                v = ",".join(v)
            query_params_list.append(f"{k}={v}")
        query_params = "&".join(query_params_list)
        full_url = f"{url}?{query_params}"
        return full_url
