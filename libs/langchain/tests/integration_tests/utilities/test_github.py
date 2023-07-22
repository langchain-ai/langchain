"""Integration test for Github Wrapper."""
import pytest

from langchain.utilities.github import GitHubAPIWrapper

# Make sure you have set the following env variables:
# GITHUB_REPOSITORY
# GITHUB_BRANCH
# GITHUB_APP_ID
# GITHUB_PRIVATE_KEY


@pytest.fixture
def api_client() -> GitHubAPIWrapper:
    return GitHubAPIWrapper()


def test_get_open_issues(api_client: GitHubAPIWrapper) -> None:
    """Basic test to fetch issues"""
    issues = api_client.get_issues()
    assert len(issues) != 0
