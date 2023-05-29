import pytest
from pytest_mock import MockerFixture

from langchain.docstore.document import Document
from langchain.document_loaders.github import GitHubLoader


def test_initialization() -> None:
    loader = GitHubLoader(repo="repo", access_token="access_token")
    assert loader.repo == "repo"
    assert loader.access_token == "access_token"
    assert loader.headers == {
        "Accept": "application/vnd.github+json",
        "Authorization": "Bearer access_token",
    }


def test_load(mocker: MockerFixture) -> None:
    mocker.patch("requests.get", return_value=mocker.MagicMock(json=lambda: []))
    loader = GitHubLoader(repo="repo", access_token="access_token")
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
            "creation_time": issue["created_at"],
            "comments": issue["comments"],
            "state": issue["state"],
            "labels": [label["name"] for label in issue["labels"]],  # type: ignore
            "assignee": issue["assignee"]["login"],  # type: ignore
            "milestone": issue["milestone"]["title"],  # type: ignore
            "locked": issue["locked"],
            "number": issue["number"],
        },
    )
    loader = GitHubLoader(repo="repo", access_token="access_token")
    document = loader.parse_issue(issue)
    assert document == expected_document


def test_build_url() -> None:
    loader = GitHubLoader(repo="repo", access_token="access_token")

    # No parameters
    url = loader.build_url()
    assert url == "https://api.github.com/repos/repo/issues?"

    # parameters: state,  sort
    url = loader.build_url(state="open", sort="created")
    assert url == "https://api.github.com/repos/repo/issues?state=open&sort=created"

    # parameters: milestone, state, assignee, creator, mentioned, labels, sort,
    # direction, since
    url = loader.build_url(
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
    assert url == (
        "https://api.github.com/repos/repo/issues?milestone=*&state=closed"
        "&assignee=user1&creator=user2&mentioned=user3&labels=bug,ui,@high"
        "&sort=comments&direction=asc&since=2023-05-26T00:00:00Z"
    )

    # Invalid parameter
    with pytest.raises(ValueError):
        loader.build_url(invalid="parameter")

    # Invalid value for valid parameter
    with pytest.raises(ValueError):
        loader.build_url(state="invalid_state")

    # Invalid type for labels
    with pytest.raises(ValueError):
        loader.build_url(labels="not_a_list")

    # Invalid date format for since
    with pytest.raises(ValueError):
        loader.build_url(since="not_a_date")


def test_integration() -> None:
    title = (
        "ChatOpenAI models don't work with prompts created via ChatPromptTemplate."
        "from_role_strings"
    )
    loader = GitHubLoader(repo="hwchase17/langchain")
    data = loader.load(creator="UmerHA")
    titles = [d.metadata["title"] for d in data]
    assert title in titles
