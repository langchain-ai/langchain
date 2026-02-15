"""Unit tests for metadata.user_id handling in AnthropicLLM."""

from unittest.mock import MagicMock

from langchain_anthropic._client_utils import extract_metadata_from_run_manager


def test_extract_metadata_from_run_manager_with_user_id() -> None:
    """Test that extract_metadata_from_run_manager extracts user_id correctly."""
    mock_run_manager = MagicMock()
    mock_run_manager.metadata = {"user_id": "test-user-123", "other_field": "value"}

    result = extract_metadata_from_run_manager(mock_run_manager)

    assert result is not None
    assert result == {"user_id": "test-user-123"}


def test_extract_metadata_from_run_manager_without_user_id() -> None:
    """Test that extract_metadata_from_run_manager returns None when user_id is absent."""
    mock_run_manager = MagicMock()
    mock_run_manager.metadata = {"other_field": "value"}

    result = extract_metadata_from_run_manager(mock_run_manager)

    assert result is None


def test_extract_metadata_from_run_manager_with_none() -> None:
    """Test that extract_metadata_from_run_manager handles None run_manager."""
    result = extract_metadata_from_run_manager(None)

    assert result is None


def test_extract_metadata_from_run_manager_without_metadata_attr() -> None:
    """Test that extract_metadata_from_run_manager handles run_manager without metadata."""
    mock_run_manager = MagicMock(spec=[])  # No attributes

    result = extract_metadata_from_run_manager(mock_run_manager)

    assert result is None
