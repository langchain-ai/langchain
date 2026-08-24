"""Unit tests for langchain_decodo tools.

All tests run without any network access — httpx calls are mocked via
``unittest.mock``. No ``DECODO_API_TOKEN`` environment variable is required.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from langchain_decodo.tools import (
    DecodoSearchTool,
    DecodoWebScrapeTool,
    _ENGINE_TARGET_MAP,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_api_response(content: str = "Hello world") -> dict[str, Any]:
    """Return a minimal Decodo API response dict."""
    return {
        "results": [
            {
                "content": content,
                "status_code": 200,
                "url": "https://example.com",
            }
        ]
    }


def _make_mock_httpx_response(
    json_data: dict[str, Any], status_code: int = 200
) -> MagicMock:
    """Build a mock that behaves like an httpx.Response."""
    mock_resp = MagicMock()
    mock_resp.is_success = status_code < 400
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    return mock_resp


# ---------------------------------------------------------------------------
# DecodoWebScrapeTool — initialisation
# ---------------------------------------------------------------------------


class TestDecodoWebScrapeToolInit:
    def test_explicit_token(self) -> None:
        tool = DecodoWebScrapeTool(decodo_api_token=SecretStr("tok123"))
        assert tool.decodo_api_token.get_secret_value() == "tok123"

    def test_env_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DECODO_API_TOKEN", "env_tok")
        tool = DecodoWebScrapeTool()
        assert tool.decodo_api_token.get_secret_value() == "env_tok"

    def test_name_and_description(self) -> None:
        tool = DecodoWebScrapeTool(decodo_api_token=SecretStr("x"))
        assert tool.name == "decodo_scrape_url"
        assert "URL" in tool.description or "url" in tool.description.lower()

    def test_default_base_url(self) -> None:
        tool = DecodoWebScrapeTool(decodo_api_token=SecretStr("x"))
        assert tool.base_url == "https://scraper-api.decodo.com"

    def test_custom_base_url(self) -> None:
        tool = DecodoWebScrapeTool(
            decodo_api_token=SecretStr("x"),
            base_url="https://custom.example.com",
        )
        assert tool.base_url == "https://custom.example.com"

    def test_missing_token_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DECODO_API_TOKEN", raising=False)
        tool = DecodoWebScrapeTool()
        with pytest.raises(ValueError, match="Decodo API token"):
            tool._run("https://example.com")


# ---------------------------------------------------------------------------
# DecodoWebScrapeTool — _run
# ---------------------------------------------------------------------------


class TestDecodoWebScrapeToolRun:
    @patch("langchain_decodo.tools.httpx.post")
    def test_successful_scrape(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _make_mock_httpx_response(
            _make_api_response("Scraped content here")
        )
        tool = DecodoWebScrapeTool(decodo_api_token=SecretStr("tok"))
        result = tool._run("https://example.com")
        assert result == "Scraped content here"

    @patch("langchain_decodo.tools.httpx.post")
    def test_request_includes_correct_payload(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _make_mock_httpx_response(_make_api_response())
        tool = DecodoWebScrapeTool(decodo_api_token=SecretStr("tok"))
        tool._run("https://target.com/page")

        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert payload["target"] == "universal"
        assert payload["url"] == "https://target.com/page"

    @patch("langchain_decodo.tools.httpx.post")
    def test_authorization_header(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _make_mock_httpx_response(_make_api_response())
        tool = DecodoWebScrapeTool(decodo_api_token=SecretStr("mytoken"))
        tool._run("https://example.com")

        _, kwargs = mock_post.call_args
        assert kwargs["headers"]["Authorization"] == "Basic mytoken"

    @patch("langchain_decodo.tools.httpx.post")
    def test_empty_results_returns_fallback(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _make_mock_httpx_response({"results": []})
        tool = DecodoWebScrapeTool(decodo_api_token=SecretStr("tok"))
        result = tool._run("https://example.com")
        assert result == "(No content returned by Decodo API)"

    @patch("langchain_decodo.tools.httpx.post")
    def test_api_error_raises_runtime_error(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _make_mock_httpx_response(
            {"message": "Unauthorized"}, status_code=401
        )
        tool = DecodoWebScrapeTool(decodo_api_token=SecretStr("bad"))
        with pytest.raises(RuntimeError, match="Decodo API error"):
            tool._run("https://example.com")

    @patch("langchain_decodo.tools.httpx.post")
    def test_structured_content_is_serialised(self, mock_post: MagicMock) -> None:
        structured = {"title": "Hello", "items": [1, 2, 3]}
        mock_post.return_value = _make_mock_httpx_response(
            {"results": [{"content": structured, "status_code": 200}]}
        )
        tool = DecodoWebScrapeTool(decodo_api_token=SecretStr("tok"))
        result = tool._run("https://example.com")
        parsed = json.loads(result)
        assert parsed["title"] == "Hello"

    @patch("langchain_decodo.tools.httpx.post")
    def test_network_timeout_raises_runtime_error(self, mock_post: MagicMock) -> None:
        import httpx as _httpx

        mock_post.side_effect = _httpx.TimeoutException("timed out")
        tool = DecodoWebScrapeTool(decodo_api_token=SecretStr("tok"))
        with pytest.raises(RuntimeError, match="timed out"):
            tool._run("https://example.com")


# ---------------------------------------------------------------------------
# DecodoSearchTool — initialisation
# ---------------------------------------------------------------------------


class TestDecodoSearchToolInit:
    def test_explicit_token(self) -> None:
        tool = DecodoSearchTool(decodo_api_token=SecretStr("tok"))
        assert tool.decodo_api_token.get_secret_value() == "tok"

    def test_env_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DECODO_API_TOKEN", "env_tok")
        tool = DecodoSearchTool()
        assert tool.decodo_api_token.get_secret_value() == "env_tok"

    def test_name_and_description(self) -> None:
        tool = DecodoSearchTool(decodo_api_token=SecretStr("x"))
        assert tool.name == "decodo_search"
        assert "search" in tool.description.lower()

    def test_default_engine(self) -> None:
        tool = DecodoSearchTool(decodo_api_token=SecretStr("x"))
        assert tool.engine == "google"

    def test_missing_token_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DECODO_API_TOKEN", raising=False)
        tool = DecodoSearchTool()
        with pytest.raises(ValueError, match="Decodo API token"):
            tool._run("python web scraping")


# ---------------------------------------------------------------------------
# DecodoSearchTool — _run
# ---------------------------------------------------------------------------


class TestDecodoSearchToolRun:
    @patch("langchain_decodo.tools.httpx.post")
    def test_google_search_returns_json(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _make_mock_httpx_response(
            {
                "results": [
                    {
                        "content": {"organic": []},
                        "status_code": 200,
                        "url": "",
                    }
                ]
            }
        )
        tool = DecodoSearchTool(decodo_api_token=SecretStr("tok"))
        result = tool._run("python tips", engine="google")
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 1

    @patch("langchain_decodo.tools.httpx.post")
    def test_google_engine_maps_to_correct_target(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _make_mock_httpx_response({"results": []})
        tool = DecodoSearchTool(decodo_api_token=SecretStr("tok"))
        tool._run("something", engine="google")

        _, kwargs = mock_post.call_args
        assert kwargs["json"]["target"] == "google_search"

    @patch("langchain_decodo.tools.httpx.post")
    def test_amazon_engine_maps_to_correct_target(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _make_mock_httpx_response({"results": []})
        tool = DecodoSearchTool(decodo_api_token=SecretStr("tok"))
        tool._run("laptop", engine="amazon")

        _, kwargs = mock_post.call_args
        assert kwargs["json"]["target"] == "amazon_search"

    @patch("langchain_decodo.tools.httpx.post")
    def test_reddit_engine_prepends_site_filter(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _make_mock_httpx_response({"results": []})
        tool = DecodoSearchTool(decodo_api_token=SecretStr("tok"))
        tool._run("best books", engine="reddit")

        _, kwargs = mock_post.call_args
        assert "site:reddit.com" in kwargs["json"]["query"]
        assert kwargs["json"]["target"] == "google_search"

    @patch("langchain_decodo.tools.httpx.post")
    def test_unknown_engine_defaults_to_google(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _make_mock_httpx_response({"results": []})
        tool = DecodoSearchTool(decodo_api_token=SecretStr("tok"))
        tool._run("query", engine="bing")

        _, kwargs = mock_post.call_args
        assert kwargs["json"]["target"] == "google_search"

    @patch("langchain_decodo.tools.httpx.post")
    def test_num_results_forwarded(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _make_mock_httpx_response({"results": []})
        tool = DecodoSearchTool(decodo_api_token=SecretStr("tok"))
        tool._run("query", num_results=5)

        _, kwargs = mock_post.call_args
        assert kwargs["json"]["limit"] == 5

    @patch("langchain_decodo.tools.httpx.post")
    def test_empty_results_returns_empty_json_list(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _make_mock_httpx_response({"results": []})
        tool = DecodoSearchTool(decodo_api_token=SecretStr("tok"))
        result = tool._run("nothing")
        assert json.loads(result) == []

    @patch("langchain_decodo.tools.httpx.post")
    def test_api_error_raises_runtime_error(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _make_mock_httpx_response(
            {"message": "Rate limited"}, status_code=429
        )
        tool = DecodoSearchTool(decodo_api_token=SecretStr("tok"))
        with pytest.raises(RuntimeError, match="Decodo API error"):
            tool._run("query")

    @patch("langchain_decodo.tools.httpx.post")
    def test_authorization_header(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _make_mock_httpx_response({"results": []})
        tool = DecodoSearchTool(decodo_api_token=SecretStr("secret"))
        tool._run("query")

        _, kwargs = mock_post.call_args
        assert kwargs["headers"]["Authorization"] == "Basic secret"

    def test_engine_target_map_coverage(self) -> None:
        """All documented engines must appear in the target map."""
        for engine in ("google", "amazon", "reddit"):
            assert engine in _ENGINE_TARGET_MAP


# ---------------------------------------------------------------------------
# Integration between schema and _run
# ---------------------------------------------------------------------------


class TestArgsSchema:
    def test_scrape_schema_accepts_url(self) -> None:
        tool = DecodoWebScrapeTool(decodo_api_token=SecretStr("x"))
        schema = tool.args_schema
        assert schema is not None
        instance = schema(url="https://example.com")
        assert instance.url == "https://example.com"

    def test_search_schema_defaults(self) -> None:
        tool = DecodoSearchTool(decodo_api_token=SecretStr("x"))
        schema = tool.args_schema
        assert schema is not None
        instance = schema(query="hello")
        assert instance.engine == "google"
        assert instance.num_results == 10
