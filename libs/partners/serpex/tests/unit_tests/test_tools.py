"""Tests for SERPEX Search Tool."""

import os
from typing import Any

import pytest
from pydantic import SecretStr

from langchain_serpex import SerpexSearchResults


def test_serpex_initialization() -> None:
    """Test that Serpex can be initialized with API key."""
    tool = SerpexSearchResults(api_key=SecretStr("test_api_key_12345"))
    assert tool.name == "serpex_search"
    assert tool.engine == "auto"
    assert tool.category == "web"


def test_serpex_initialization_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Serpex can be initialized from environment variable."""
    monkeypatch.setenv("SERPEX_API_KEY", "env_test_key")
    tool = SerpexSearchResults()
    assert tool.name == "serpex_search"


def test_serpex_custom_parameters() -> None:
    """Test that Serpex accepts custom parameters."""
    tool = SerpexSearchResults(
        api_key=SecretStr("test_api_key"),
        engine="google",
        category="web",
        time_range="day",
    )
    assert tool.engine == "google"
    assert tool.time_range == "day"


def test_serpex_build_params() -> None:
    """Test that _build_params creates correct parameters."""
    tool = SerpexSearchResults(
        api_key=SecretStr("test_key"), engine="bing", time_range="week"
    )
    params = tool._build_params("test query")

    assert params["q"] == "test query"
    assert params["engine"] == "bing"
    assert params["category"] == "web"
    assert params["time_range"] == "week"


def test_serpex_build_params_override() -> None:
    """Test that _build_params allows overrides."""
    tool = SerpexSearchResults(api_key=SecretStr("test_key"), engine="google")
    params = tool._build_params("test query", engine="duckduckgo", time_range="month")

    assert params["engine"] == "duckduckgo"
    assert params["time_range"] == "month"


def test_serpex_format_results_with_organic() -> None:
    """Test formatting organic search results."""
    tool = SerpexSearchResults(api_key=SecretStr("test_key"))

    mock_data: dict[str, Any] = {
        "metadata": {"number_of_results": 2},
        "results": [
            {
                "position": 1,
                "title": "Test Result 1",
                "url": "https://example.com/1",
                "snippet": "This is a test snippet",
                "published_date": None,
            },
            {
                "position": 2,
                "title": "Test Result 2",
                "url": "https://example.com/2",
                "snippet": "Another test snippet",
                "published_date": "2025-01-22",
            },
        ],
    }

    formatted = tool._format_results(mock_data)

    assert "Found 2 results" in formatted
    assert "Test Result 1" in formatted
    assert "Test Result 2" in formatted
    assert "https://example.com/1" in formatted
    assert "This is a test snippet" in formatted
    assert "Published: 2025-01-22" in formatted


def test_serpex_format_results_with_answers() -> None:
    """Test formatting instant answers."""
    tool = SerpexSearchResults(api_key=SecretStr("test_key"))

    mock_data = {
        "answers": [{"answer": "The capital of France is Paris."}],
        "results": [],
    }

    formatted = tool._format_results(mock_data)
    assert "Answer: The capital of France is Paris." in formatted


def test_serpex_format_results_with_infoboxes() -> None:
    """Test formatting knowledge panel/infobox."""
    tool = SerpexSearchResults(api_key=SecretStr("test_key"))

    mock_data = {
        "infoboxes": [{"description": "Python is a high-level programming language."}],
        "results": [],
    }

    formatted = tool._format_results(mock_data)
    assert "Knowledge Panel: Python is a high-level programming language." in formatted


def test_serpex_format_results_with_suggestions() -> None:
    """Test formatting search suggestions."""
    tool = SerpexSearchResults(api_key=SecretStr("test_key"))

    mock_data = {
        "results": [],
        "suggestions": ["coffee shops near me open now", "best coffee shops downtown"],
    }

    formatted = tool._format_results(mock_data)
    assert "Related searches" in formatted
    assert "coffee shops near me open now" in formatted


def test_serpex_format_results_with_corrections() -> None:
    """Test formatting query corrections."""
    tool = SerpexSearchResults(api_key=SecretStr("test_key"))

    mock_data = {
        "results": [],
        "corrections": ["python programming", "python language"],
    }

    formatted = tool._format_results(mock_data)
    assert "Did you mean" in formatted
    assert "python programming" in formatted


def test_serpex_format_results_empty() -> None:
    """Test formatting when no results."""
    tool = SerpexSearchResults(api_key=SecretStr("test_key"))

    mock_data: dict[str, Any] = {"results": []}

    formatted = tool._format_results(mock_data)
    assert formatted == "No search results found."


def test_serpex_custom_base_url() -> None:
    """Test that custom base URL is respected."""
    custom_url = "https://custom-api.example.com"
    tool = SerpexSearchResults(api_key=SecretStr("test_key"), base_url=custom_url)
    assert tool.base_url == custom_url


# Integration test (requires real API key)
@pytest.mark.skipif("SERPEX_API_KEY" not in os.environ, reason="SERPEX_API_KEY not set")
def test_serpex_real_search() -> None:
    """Test with real API (requires SERPEX_API_KEY environment variable)."""
    api_key = os.getenv("SERPEX_API_KEY")

    if not api_key:
        pytest.skip("SERPEX_API_KEY not set")

    tool = SerpexSearchResults(
        api_key=SecretStr(api_key), engine="auto", category="web"
    )

    result = tool._run("weather in San Francisco")

    assert result
    assert len(result) > 0
    assert "Error" not in result or "No search results" in result
