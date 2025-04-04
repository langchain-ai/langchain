"""Integration test for Github Wrapper."""

import pytest

from langchain_community.utilities.github import GitHubAPIWrapper

# Make sure you have set the following env variables:
# GITHUB_REPOSITORY
# GITHUB_BRANCH
# GITHUB_APP_ID
# GITHUB_PRIVATE_KEY


@pytest.fixture
def api_client() -> GitHubAPIWrapper:
    return GitHubAPIWrapper()  # type: ignore[call-arg]


def test_get_open_issues(api_client: GitHubAPIWrapper) -> None:
    """Basic test to fetch issues"""
    issues = api_client.get_issues()
    assert len(issues) != 0


def test_get_latest_release(api_client: GitHubAPIWrapper) -> None:
    """Basic test to fetch latest release"""
    release = api_client.get_latest_release()
    assert release is not None


def test_get_releases(api_client: GitHubAPIWrapper) -> None:
    """Basic test to fetch releases"""
    releases = api_client.get_releases()
    assert releases is not None


def test_search_issues_and_prs(api_client: GitHubAPIWrapper) -> None:
    """Basic test to search issues and PRs"""
    results = api_client.search_issues_and_prs("is:pr is:merged")
    assert len(results) != 0
