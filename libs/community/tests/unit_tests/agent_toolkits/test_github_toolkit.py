from unittest.mock import MagicMock

from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper


def test_github_toolkit() -> None:
    # Create a mock GitHub wrapper with required attributes
    mock_github = MagicMock(spec=GitHubAPIWrapper)
    mock_github.github_repository = "fake/repo"
    mock_github.github_app_id = "fake_id"
    mock_github.github_app_private_key = "fake_key"
    mock_github.active_branch = "main"
    mock_github.github_base_branch = "main"

    # Test without release tools
    toolkit = GitHubToolkit.from_github_api_wrapper(mock_github)
    tools = toolkit.get_tools()
    assert len(tools) == 21  # Base number of tools

    # Test with release tools
    toolkit_with_releases = GitHubToolkit.from_github_api_wrapper(
        mock_github, include_release_tools=True
    )
    tools_with_releases = toolkit_with_releases.get_tools()
    assert len(tools_with_releases) == 24  # Base tools + 3 release tools
