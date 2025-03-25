import base64

import pytest
from langchain_core.documents import Document
from pytest_mock import MockerFixture

from langchain_community.document_loaders.github import (
    GithubFileLoader,
    GitHubIssuesLoader,
)


def test_initialization() -> None:
    loader = GitHubIssuesLoader(repo="repo", access_token="access_token")
    assert loader.repo == "repo"
    assert loader.access_token == "access_token"
    assert loader.headers == {
        "Accept": "application/vnd.github+json",
        "Authorization": "Bearer access_token",
    }


def test_initialization_ghe() -> None:
    loader = GitHubIssuesLoader(
        repo="repo",
        access_token="access_token",
        github_api_url="https://github.example.com/api/v3",
    )
    assert loader.repo == "repo"
    assert loader.access_token == "access_token"
    assert loader.github_api_url == "https://github.example.com/api/v3"
    assert loader.headers == {
        "Accept": "application/vnd.github+json",
        "Authorization": "Bearer access_token",
    }


def test_invalid_initialization() -> None:
    # Invalid parameter
    with pytest.raises(ValueError):
        GitHubIssuesLoader(invalid="parameter")  # type: ignore[call-arg]

    # Invalid value for valid parameter
    with pytest.raises(ValueError):
        GitHubIssuesLoader(state="invalid_state")  # type: ignore[arg-type, call-arg]

    # Invalid type for labels
    with pytest.raises(ValueError):
        GitHubIssuesLoader(labels="not_a_list")  # type: ignore[arg-type, call-arg]

    # Invalid date format for since
    with pytest.raises(ValueError):
        GitHubIssuesLoader(since="not_a_date")  # type: ignore[call-arg]


def test_load_github_issue(mocker: MockerFixture) -> None:
    mocker.patch(
        "requests.get", return_value=mocker.MagicMock(json=lambda: [], links=None)
    )
    loader = GitHubIssuesLoader(repo="repo", access_token="access_token")
    documents = loader.load()
    assert documents == []


def test_parse_issue() -> None:
    issue = {
        "html_url": "https://github.com/repo/issue/1",
        "title": "Example Issue 1",
        "user": {"login": "username1"},
        "created_at": "2023-01-01T00:00:00Z",
        "comments": 1,
        "state": "open",
        "labels": [{"name": "bug"}],
        "assignee": {"login": "username2"},
        "milestone": {"title": "v1.0"},
        "locked": "False",
        "number": "1",
        "body": "This is an example issue 1",
    }
    expected_document = Document(
        page_content=issue["body"],  # type: ignore
        metadata={
            "url": issue["html_url"],
            "title": issue["title"],
            "creator": issue["user"]["login"],  # type: ignore
            "created_at": issue["created_at"],
            "comments": issue["comments"],
            "state": issue["state"],
            "labels": [label["name"] for label in issue["labels"]],  # type: ignore
            "assignee": issue["assignee"]["login"],  # type: ignore
            "milestone": issue["milestone"]["title"],  # type: ignore
            "locked": issue["locked"],
            "number": issue["number"],
            "is_pull_request": False,
        },
    )
    loader = GitHubIssuesLoader(repo="repo", access_token="access_token")
    document = loader.parse_issue(issue)
    assert document == expected_document


def test_url() -> None:
    # No parameters
    loader = GitHubIssuesLoader(repo="repo", access_token="access_token")
    assert loader.url == "https://api.github.com/repos/repo/issues?"

    # parameters: state,  sort
    loader = GitHubIssuesLoader(
        repo="repo", access_token="access_token", state="open", sort="created"
    )
    assert (
        loader.url == "https://api.github.com/repos/repo/issues?state=open&sort=created"
    )

    # parameters: milestone, state, assignee, creator, mentioned, labels, sort,
    # direction, since
    loader = GitHubIssuesLoader(
        repo="repo",
        access_token="access_token",
        milestone="*",
        state="closed",
        assignee="user1",
        creator="user2",
        mentioned="user3",
        labels=["bug", "ui", "@high"],
        sort="comments",
        direction="asc",
        since="2023-05-26T00:00:00Z",
    )
    assert loader.url == (
        "https://api.github.com/repos/repo/issues?milestone=*&state=closed"
        "&assignee=user1&creator=user2&mentioned=user3&labels=bug,ui,@high"
        "&sort=comments&direction=asc&since=2023-05-26T00:00:00Z"
    )


def test_github_file_content_get_file_paths(mocker: MockerFixture) -> None:
    # Mock the requests.get method to simulate the API response
    mocker.patch(
        "requests.get",
        return_value=mocker.MagicMock(
            json=lambda: {
                "tree": [
                    {
                        "path": "readme.md",
                        "mode": "100644",
                        "type": "blob",
                        "sha": "789",
                        "size": 37,
                        "url": "https://github.com/repos/shufanhao/langchain/git/blobs/789",
                    }
                ]
            },
            status_code=200,
        ),
    )

    # case1: add file_filter
    loader = GithubFileLoader(
        repo="shufanhao/langchain",
        access_token="access_token",
        github_api_url="https://github.com",
        file_filter=lambda file_path: file_path.endswith(".md"),
    )

    # Call the load method
    files = loader.get_file_paths()

    # Assert the results
    assert len(files) == 1
    assert files[0]["path"] == "readme.md"

    # case2: didn't add file_filter
    loader = GithubFileLoader(
        repo="shufanhao/langchain",
        access_token="access_token",
        github_api_url="https://github.com",
        file_filter=None,
    )

    # Call the load method
    files = loader.get_file_paths()
    assert len(files) == 1
    assert files[0]["path"] == "readme.md"

    # case3: add file_filter with a non-exist file path
    loader = GithubFileLoader(
        repo="shufanhao/langchain",
        access_token="access_token",
        github_api_url="https://github.com",
        file_filter=lambda file_path: file_path.endswith(".py"),
    )

    # Call the load method
    files = loader.get_file_paths()
    assert len(files) == 0


def test_github_file_content_loader(mocker: MockerFixture) -> None:
    # Mock the requests.get method to simulate the API response
    file_path_res = mocker.MagicMock(
        json=lambda: {
            "tree": [
                {
                    "path": "readme.md",
                    "mode": "100644",
                    "type": "blob",
                    "sha": "789",
                    "size": 37,
                    "url": "https://github.com/repos/shufanhao/langchain/git/blobs/789",
                }
            ]
        },
        status_code=200,
    )
    file_content_res = mocker.MagicMock(
        json=lambda: {"content": base64.b64encode("Mocked content".encode("utf-8"))},
        status_code=200,
    )

    mocker.patch("requests.get", side_effect=[file_path_res, file_content_res])

    # case1: file_extension=".md"
    loader = GithubFileLoader(
        repo="shufanhao/langchain",
        access_token="access_token",
        github_api_url="https://github.com",
        file_filter=None,
    )

    # Call the load method
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].page_content == "Mocked content"
    assert docs[0].metadata["sha"] == "789"
