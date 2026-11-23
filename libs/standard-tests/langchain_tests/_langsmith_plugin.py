"""Pytest plugin that forwards CI-set LangSmith env vars into tracing context.

The LangSmith SDK does not natively read `LANGSMITH_TAGS` / `LANGSMITH_METADATA`
from the environment, so tags/metadata written to `$GITHUB_ENV` by CI workflows
(e.g. `integration_tests.yml`) would otherwise be silently dropped. This plugin
bridges that gap for the duration of the pytest session by entering
`langsmith.run_helpers.tracing_context`.

To avoid surprising developers who happen to export these vars locally, the
plugin only activates when running under GitHub Actions (`GITHUB_ACTIONS=true`).

Auto-discovered by pytest in any package that depends on `langchain-tests`
(declared via the `pytest11` entry point in
`libs/standard-tests/pyproject.toml`).
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import warnings
from typing import TYPE_CHECKING, Any

import pytest
from langsmith.run_helpers import tracing_context

if TYPE_CHECKING:
    from collections.abc import Iterator


def _is_github_actions() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true"


def _parse_tags(raw: str) -> list[str]:
    return [tag.strip() for tag in raw.split(",") if tag.strip()]


def _parse_metadata(raw: str) -> dict[str, Any] | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        _warn_loud(f"Ignoring LANGSMITH_METADATA: invalid JSON ({exc}).")
        return None
    if not isinstance(parsed, dict):
        _warn_loud("Ignoring LANGSMITH_METADATA: expected a JSON object.")
        return None
    return parsed


def _warn_loud(message: str) -> None:
    """Emit a `UserWarning` and mirror it to stderr so CI logs surface it."""
    warnings.warn(message, UserWarning, stacklevel=3)
    sys.stderr.write(f"[langchain-tests] {message}\n")


@contextlib.contextmanager
def _langsmith_ci_cm() -> Iterator[None]:
    """Yield with tracing context applied from current env vars.

    No-op when neither env var is set or when not running on GitHub Actions.
    """
    if not _is_github_actions():
        yield
        return

    tags = _parse_tags(os.environ.get("LANGSMITH_TAGS", ""))
    metadata = _parse_metadata(os.environ.get("LANGSMITH_METADATA", ""))

    if not tags and not metadata:
        yield
        return

    keys = sorted(metadata or {})
    sys.stderr.write(
        f"[langchain-tests] langsmith CI context: tags={tags}, metadata_keys={keys}\n",
    )
    with tracing_context(tags=tags or None, metadata=metadata):
        yield


@pytest.fixture(scope="session", autouse=True)
def _langsmith_ci_context() -> Iterator[None]:
    """Apply `LANGSMITH_TAGS`/`LANGSMITH_METADATA` to traces in this session."""
    with _langsmith_ci_cm():
        yield
