import requests
from typing import List, Any
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class GitHubLoader(BaseLoader):
    """Load GitHub repo data via the GitHub REST API."""

    def __init__(self, repo: str, access_token: str):
        """Initialize GitHubLoader with repo and access_token."""
        self.repo = repo
        self.access_token = access_token
        self.headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.access_token}",
        }

    def load(self, **kwargs: Any) -> List[Document]:
        """Get important GitHub repository information.

        Args:
            filter: Can be one of assigned, created, mentioned, subscribed, repos, all
            state: Can be onf of open, closed, all
            labels: A list of comma separated label names. Example: bug,ui,@high
            collab boolean
            orgs boolean
            owned boolean
            pulls boolean
            sort - Can be one of created, updated, comments
            desc - asc, desc
            since - YYYY-MM-DDTHH:MM:SSZ

        Returns a  are:
            - title
            - content
            - url,
            - creation time
            - creator of the issue
            - number of comments
        """
        url = f"https://api.github.com/repos/{self.repo}/issues"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        issues = response.json()
        return [self.parse_issue(issue) for issue in issues]

    def build_url(self, **kwargs: Any) -> str:
        valid_kwargs = {
            "filter": {
                "assigned",
                "created",
                "mentioned",
                "subscribed",
                "repos",
                "all",
            },
            "state": {"open", "closed", "all"},
            "sort": {"created", "updated", "comments"},
            "desc": {"asc", "desc"},
            "collab": {True, False},
            "orgs": {True, False},
            "owned": {True, False},
            "pulls": {True, False},
        }

        for key, value in kwargs.items():
            if key in ["labels", "since"]:
                # we validate "labels" and "since" below
                continue

            if key not in valid_kwargs:
                raise ValueError(f"Invalid keyword argument: {key}")
            if value not in valid_kwargs[key]:
                raise ValueError(
                    f"Invalid value for {key}: {value}. Expected one of {valid_kwargs[key]}"
                )

        if "labels" in kwargs and not isinstance(kwargs["labels"], list):
            raise ValueError("Invalid value for labels: Expected a list of strings")

        if "since" in kwargs:
            try:
                datetime.datetime.strptime(kwargs["since"], "%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                raise ValueError(
                    "Invalid value for since: Expected a date string in YYYY-MM-DDTHH:MM:SSZ format"
                )

        return "todo"

    def load_filtered(self, query: str) -> List[Document]:
        """Search for GitHub issues based on a query."""
        url = f"https://api.github.com/search/issues"
        params = {"q": f"repo:{self.repo} {query}"}
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()

        issues = response.json()["items"]
        return [self.parse_issue(issue) for issue in issues]

    def parse_issue(self, issue: dict) -> Document:
        """Create Document objects from a list of GitHub issues."""
        metadata = {
            "url": issue["html_url"],
            "title": issue["title"],
            "creator": issue["user"]["login"],
            "creation_time": issue["created_at"],
            "comments": issue["comments"],
        }
        return Document(page_content=issue["body"], metadata=metadata)
