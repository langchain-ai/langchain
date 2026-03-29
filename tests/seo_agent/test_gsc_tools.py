"""Tests for Google Search Console tools — connection and data retrieval."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

os.environ["AHREFS_MOCK"] = "true"
os.environ["GSC_MOCK"] = "true"
os.environ["SUPABASE_MOCK"] = "true"
os.environ["TAVILY_MOCK"] = "true"
os.environ["ANTHROPIC_API_KEY"] = "mock-key-for-testing"

from agents.seo_agent.tools.gsc_tools import (
    get_top_pages,
    get_top_queries,
    test_connection,
)


class TestGSCConnection:
    """Tests for the test_connection function in mock mode."""

    def test_mock_connection_returns_ok(self) -> None:
        result = test_connection()
        assert result["ok"] is True
        assert result["status"] == "mock"
        assert isinstance(result["sites"], list)
        assert len(result["sites"]) > 0

    def test_mock_connection_has_expected_sites(self) -> None:
        result = test_connection()
        assert "https://www.kitchensdirectory.co.uk" in result["sites"]
        assert "https://www.freeroomplanner.com" in result["sites"]
        assert "https://www.kitchencostestimator.com" in result["sites"]

    def test_mock_connection_detail_mentions_mock(self) -> None:
        result = test_connection()
        assert "mock" in result["detail"].lower()

    def test_connection_returns_required_keys(self) -> None:
        result = test_connection()
        assert "ok" in result
        assert "status" in result
        assert "detail" in result
        assert "sites" in result


class TestGSCConnectionMissingCredentials:
    """Tests that simulate real mode without credentials."""

    def setup_method(self) -> None:
        self._original_mock = os.environ.get("GSC_MOCK")
        self._original_path = os.environ.get("GSC_SERVICE_ACCOUNT_PATH")
        os.environ["GSC_MOCK"] = "false"
        os.environ.pop("GSC_SERVICE_ACCOUNT_PATH", None)

    def teardown_method(self) -> None:
        if self._original_mock is not None:
            os.environ["GSC_MOCK"] = self._original_mock
        else:
            os.environ.pop("GSC_MOCK", None)
        if self._original_path is not None:
            os.environ["GSC_SERVICE_ACCOUNT_PATH"] = self._original_path
        else:
            os.environ.pop("GSC_SERVICE_ACCOUNT_PATH", None)

    def test_missing_credentials_returns_error(self) -> None:
        result = test_connection()
        assert result["ok"] is False
        assert result["status"] == "error"
        assert "GSC_SERVICE_ACCOUNT_PATH" in result["detail"]
        assert result["sites"] == []


class TestGSCQueries:
    """Tests for data retrieval functions in mock mode."""

    def test_get_top_queries_returns_list(self) -> None:
        result = get_top_queries("https://www.kitchensdirectory.co.uk")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_get_top_pages_returns_list(self) -> None:
        result = get_top_pages("https://www.kitchensdirectory.co.uk")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_top_queries_have_expected_keys(self) -> None:
        result = get_top_queries("https://www.kitchensdirectory.co.uk")
        for row in result:
            assert "keys" in row
            assert "clicks" in row
            assert "impressions" in row

    def test_top_pages_have_expected_keys(self) -> None:
        result = get_top_pages("https://www.kitchensdirectory.co.uk")
        for row in result:
            assert "page" in row or "keys" in row
            assert "clicks" in row
