"""Tests for CLI functionality."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from langchain_model_profiles.cli import refresh


@pytest.fixture
def mock_models_dev_response() -> dict:
    """Create a mock response from models.dev API."""
    return {
        "anthropic": {
            "id": "anthropic",
            "name": "Anthropic",
            "models": {
                "claude-3-opus": {
                    "id": "claude-3-opus",
                    "name": "Claude 3 Opus",
                    "tool_call": True,
                    "limit": {"context": 200000, "output": 4096},
                    "modalities": {"input": ["text", "image"], "output": ["text"]},
                },
                "claude-3-sonnet": {
                    "id": "claude-3-sonnet",
                    "name": "Claude 3 Sonnet",
                    "tool_call": True,
                    "limit": {"context": 200000, "output": 4096},
                    "modalities": {"input": ["text", "image"], "output": ["text"]},
                },
            },
        },
        "openai": {
            "id": "openai",
            "name": "OpenAI",
            "models": {
                "gpt-4": {
                    "id": "gpt-4",
                    "name": "GPT-4",
                    "tool_call": True,
                    "limit": {"context": 8192, "output": 4096},
                    "modalities": {"input": ["text"], "output": ["text"]},
                }
            },
        },
    }


def test_refresh_downloads_and_saves_provider_data(
    tmp_path: Path, mock_models_dev_response: dict
) -> None:
    """Test that refresh command downloads and saves provider-specific data."""
    output_file = tmp_path / "data" / "models.json"

    # Mock the httpx.get call
    mock_response = Mock()
    mock_response.json.return_value = mock_models_dev_response
    mock_response.raise_for_status = Mock()

    with patch("langchain_model_profiles.cli.httpx.get", return_value=mock_response):
        refresh("anthropic", output_file)

    # Verify output file was created
    assert output_file.exists()

    # Verify content is correct
    with output_file.open() as f:
        saved_data = json.load(f)

    # Should only contain anthropic data
    assert len(saved_data) == 1
    assert "anthropic" in saved_data
    assert "openai" not in saved_data

    # Verify anthropic data is complete
    anthropic_data = saved_data["anthropic"]
    assert anthropic_data["id"] == "anthropic"
    assert anthropic_data["name"] == "Anthropic"
    assert len(anthropic_data["models"]) == 2
    assert "claude-3-opus" in anthropic_data["models"]
    assert "claude-3-sonnet" in anthropic_data["models"]


def test_refresh_raises_error_for_missing_provider(
    tmp_path: Path, mock_models_dev_response: dict
) -> None:
    """Test that refresh exits with error for non-existent provider."""
    output_file = tmp_path / "models.json"

    # Mock the httpx.get call
    mock_response = Mock()
    mock_response.json.return_value = mock_models_dev_response
    mock_response.raise_for_status = Mock()

    with patch("langchain_model_profiles.cli.httpx.get", return_value=mock_response):
        with pytest.raises(SystemExit) as exc_info:
            refresh("nonexistent-provider", output_file)

        assert exc_info.value.code == 1

    # Output file should not be created
    assert not output_file.exists()


def test_refresh_creates_parent_directories(
    tmp_path: Path, mock_models_dev_response: dict
) -> None:
    """Test that refresh creates parent directories if they don't exist."""
    output_file = tmp_path / "nested" / "dir" / "models.json"

    # Mock the httpx.get call
    mock_response = Mock()
    mock_response.json.return_value = mock_models_dev_response
    mock_response.raise_for_status = Mock()

    with patch("langchain_model_profiles.cli.httpx.get", return_value=mock_response):
        refresh("anthropic", output_file)

    # Verify parent directories were created
    assert output_file.parent.exists()
    assert output_file.exists()
