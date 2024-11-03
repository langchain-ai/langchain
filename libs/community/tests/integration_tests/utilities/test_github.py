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


def test_search_issues_and_prs(api_client: GitHubAPIWrapper) -> None:
    """Basic test to search issues and PRs"""
    results = api_client.search_issues_and_prs("is:pr is:merged")
    assert len(results) != 0
