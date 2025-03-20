from unittest.mock import MagicMock, patch

import pytest

from langchain_community.utilities.jira import JiraAPIWrapper


@pytest.fixture
def mock_jira():  # type: ignore
    with patch("atlassian.Jira") as mock_jira:
        yield mock_jira


@pytest.mark.requires("atlassian")
class TestJiraAPIWrapper:
    def test_jira_api_wrapper(self, mock_jira: MagicMock) -> None:
        """Test for Jira API Wrapper using mocks"""
        # Configure the mock instance
        mock_jira_instance = mock_jira.return_value

        # Mock projects method to return mock projects
        mock_project1 = MagicMock(key="PROJ1")
        mock_project2 = MagicMock(key="PROJ2")

        # Set up the mock to return our mock projects
        mock_jira_instance.projects.return_value = [mock_project1, mock_project2]

        # Initialize wrapper with mocks in place
        jira_wrapper = JiraAPIWrapper(
            jira_username="test_user",
            jira_api_token="test_token",
            jira_instance_url="https://test.atlassian.net",
            jira_cloud=True,
        )

        mock_jira.assert_called_once_with(
            url="https://test.atlassian.net",
            username="test_user",
            password="test_token",
            cloud=True,
        )

        # Test get_projects function
        result = jira_wrapper.run("get_projects", "")

        # Verify the mock was called and the result contains expected info
        mock_jira_instance.projects.assert_called_once()
        assert result.startswith("Found 2 projects")

    def test_jira_api_wrapper_with_cloud_false(self, mock_jira: MagicMock) -> None:
        JiraAPIWrapper(
            jira_username="test_user",
            jira_api_token="test_token",
            jira_instance_url="https://test.atlassian.net",
            jira_cloud=False,
        )
        mock_jira.assert_called_once_with(
            url="https://test.atlassian.net",
            username="test_user",
            password="test_token",
            cloud=False,
        )
