"""Tests for the CI LangSmith env-var pytest plugin."""

from __future__ import annotations

from textwrap import dedent

import pytest
from langsmith.run_helpers import get_tracing_context

from langchain_tests._langsmith_plugin import (
    _langsmith_ci_cm,
    _parse_metadata,
    _parse_tags,
)

pytest_plugins = ["pytester"]


@pytest.fixture(autouse=True)
def _force_ci_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Activate the plugin's CI-gated code paths for every test."""
    monkeypatch.setenv("GITHUB_ACTIONS", "true")


class TestParseTags:
    def test_splits_and_strips(self) -> None:
        assert _parse_tags("a, b ,c") == ["a", "b", "c"]

    def test_drops_empty_segments(self) -> None:
        assert _parse_tags(",a,,b,") == ["a", "b"]

    def test_empty_string(self) -> None:
        assert _parse_tags("") == []

    def test_whitespace_only(self) -> None:
        assert _parse_tags("  ,  ,  ") == []


class TestParseMetadata:
    def test_valid_object(self) -> None:
        assert _parse_metadata('{"sha": "abc", "n": 1}') == {"sha": "abc", "n": 1}

    def test_empty_string_returns_none(self) -> None:
        assert _parse_metadata("") is None
        assert _parse_metadata("   ") is None

    def test_invalid_json_warns_and_returns_none(self) -> None:
        with pytest.warns(UserWarning, match="invalid JSON"):
            result = _parse_metadata("{not json")
        assert result is None

    def test_non_object_warns_and_returns_none(self) -> None:
        with pytest.warns(UserWarning, match="JSON object"):
            result = _parse_metadata('["a", "b"]')
        assert result is None


class TestContextManager:
    def test_applies_tags_and_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGSMITH_TAGS", "github-actions,sha-deadbeef")
        monkeypatch.setenv("LANGSMITH_METADATA", '{"github_run_id": "42"}')
        with _langsmith_ci_cm():
            ctx = get_tracing_context()
            assert ctx["tags"] == ["github-actions", "sha-deadbeef"]
            assert ctx["metadata"] == {"github_run_id": "42"}

    def test_restores_context_on_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGSMITH_TAGS", "x")
        monkeypatch.delenv("LANGSMITH_METADATA", raising=False)
        before = get_tracing_context()
        with _langsmith_ci_cm():
            pass
        after = get_tracing_context()
        assert before["tags"] == after["tags"]
        assert before["metadata"] == after["metadata"]

    def test_no_env_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LANGSMITH_TAGS", raising=False)
        monkeypatch.delenv("LANGSMITH_METADATA", raising=False)
        before = get_tracing_context()
        with _langsmith_ci_cm():
            inside = get_tracing_context()
        assert inside["tags"] == before["tags"]
        assert inside["metadata"] == before["metadata"]

    def test_whitespace_only_tags_is_noop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LANGSMITH_TAGS", "  ,  ,  ")
        monkeypatch.delenv("LANGSMITH_METADATA", raising=False)
        before = get_tracing_context()
        with _langsmith_ci_cm():
            inside = get_tracing_context()
        assert inside["tags"] == before["tags"]

    def test_only_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LANGSMITH_TAGS", raising=False)
        monkeypatch.setenv("LANGSMITH_METADATA", '{"k": "v"}')
        with _langsmith_ci_cm():
            ctx = get_tracing_context()
            assert ctx["metadata"] == {"k": "v"}

    def test_only_tags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGSMITH_TAGS", "solo")
        monkeypatch.delenv("LANGSMITH_METADATA", raising=False)
        with _langsmith_ci_cm():
            ctx = get_tracing_context()
            assert ctx["tags"] == ["solo"]

    def test_bad_metadata_does_not_block_tags(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LANGSMITH_TAGS", "kept")
        monkeypatch.setenv("LANGSMITH_METADATA", "not-json")
        with pytest.warns(UserWarning, match="invalid JSON"), _langsmith_ci_cm():  # noqa: PT031
            ctx = get_tracing_context()
            assert ctx["tags"] == ["kept"]
            assert ctx["metadata"] is None

    def test_inactive_outside_github_actions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        monkeypatch.setenv("LANGSMITH_TAGS", "should-not-apply")
        before = get_tracing_context()
        with _langsmith_ci_cm():
            inside = get_tracing_context()
        assert inside["tags"] == before["tags"]


class TestPluginDiscovery:
    """End-to-end: the `pytest11` entry point wires up the autouse fixture."""

    def test_autouse_fixture_applies_env_in_subprocess(
        self, pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.setenv("LANGSMITH_TAGS", "discovered,from-entrypoint")
        monkeypatch.delenv("LANGSMITH_METADATA", raising=False)
        pytester.makepyfile(
            dedent("""
                from langsmith.run_helpers import get_tracing_context

                def test_tags_visible_via_autouse_fixture():
                    ctx = get_tracing_context()
                    assert ctx["tags"] == ["discovered", "from-entrypoint"]
            """),
        )
        result = pytester.runpytest_subprocess("-q")
        result.assert_outcomes(passed=1)
