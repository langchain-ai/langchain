"""Tests for the Slack directory loader"""
from pathlib import Path

from langchain.document_loaders import SlackDirectoryLoader


def test_slack_directory_loader() -> None:
    """Test Slack directory loader."""
    file_path = Path(__file__).parent.parent / "examples/slack_export.zip"
    loader = SlackDirectoryLoader(str(file_path))
    docs = loader.load()

    assert len(docs) == 5


def test_slack_directory_loader_urls() -> None:
    """Test workspace URLS are passed through in the SlackDirectoryloader."""
    file_path = Path(__file__).parent.parent / "examples/slack_export.zip"
    workspace_url = "example_workspace.com"
    loader = SlackDirectoryLoader(str(file_path), workspace_url)
    docs = loader.load()
    for doc in docs:
        assert doc.metadata["source"].startswith(workspace_url)
