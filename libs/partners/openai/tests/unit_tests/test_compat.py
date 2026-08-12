"""Tests for the `httpx` / `httpx2` compatibility shim (`_compat`)."""

import inspect
import logging

import openai
import pytest

from langchain_openai import _compat
from langchain_openai.chat_models import _client_utils


def test_module_httpx_matches_helper() -> None:
    """The import branch and the helper must not diverge (same predicate).

    Note this shares its predicate with the code under test, so it guards
    against the `if`/`elif`/`else` wiring in `_compat` being edited to
    contradict `_sdk_uses_httpx2`, not against the helper itself being wrong.
    """
    expected = "httpx2" if _compat._sdk_uses_httpx2() else "httpx"
    assert _compat.httpx.__name__ == expected


def test_only_httpx_is_exported() -> None:
    """`__all__` stays minimal; the module docstring explains why.

    The default client classes are deliberately left out, so nothing but the
    resolved `httpx` module should appear here.
    """
    assert _compat.__all__ == ["httpx"]
    assert not hasattr(_compat, "DefaultHttpxClient")


def test_default_clients_are_subclassable_classes() -> None:
    """`DefaultHttpxClient`/`DefaultAsyncHttpxClient` must stay classes.

    The wrappers in `_client_utils` subclass them. The parallel
    `DefaultHttpx2Client` names are factory functions and cannot be subclassed,
    which is why they are not used — but they are absent on the `openai>=2.45.0`
    floor, so that asymmetry is documented rather than asserted here.
    """
    assert inspect.isclass(openai.DefaultHttpxClient)
    assert inspect.isclass(openai.DefaultAsyncHttpxClient)


def test_wrappers_subclass_openai_default_clients() -> None:
    """The client wrappers extend the SDK's version-appropriate base classes."""
    assert issubclass(_client_utils._SyncHttpxClientWrapper, openai.DefaultHttpxClient)
    assert issubclass(
        _client_utils._AsyncHttpxClientWrapper, openai.DefaultAsyncHttpxClient
    )


def test_client_utils_transport_objects_use_shim_httpx() -> None:
    """Transport/config objects come from the same module the SDK client uses.

    If these were built from a different httpx than the SDK's backing library,
    an httpx transport would be handed to an httpx2 client (or vice versa).
    """
    assert _client_utils.httpx is _compat.httpx
    assert isinstance(_client_utils._DEFAULT_CONNECTION_LIMITS, _compat.httpx.Limits)


def test_selection_matches_sdk_backing_library() -> None:
    """The selection agrees with the library the SDK is demonstrably built on.

    An oracle independent of `_compat`: whichever httpx client class
    `openai.DefaultHttpxClient` inherits from is, by definition, the library our
    transport objects have to match. Unlike a version-string check, this catches
    a wrong selection under the SDK actually installed.
    """
    base = next(
        b
        for b in openai.DefaultHttpxClient.__mro__
        if b.__module__.split(".", 1)[0] in ("httpx", "httpx2")
    )
    expected = base.__module__.split(".", 1)[0]
    assert _compat.httpx.__name__ == expected
    assert _compat._sdk_uses_httpx2() is (expected == "httpx2")


def test_selection_ignores_version_string() -> None:
    """A mangled version must not change the answer while the SDK is readable.

    Regression guard for the original heuristic: build tooling that stamps
    `__version__` from a git tag can leave a `v` prefix, which an `int()` parse
    rejects. Selection then silently fell back to classic `httpx` even on
    openai 3, surfacing much later as `APIConnectionError`.
    """
    truth = _compat._sdk_uses_httpx2()
    for version in ("v3.0.0", "not-a-version", "", "2.0.0", "9.9.9"):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(openai, "__version__", version)
            assert _compat._sdk_uses_httpx2() is truth


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("2.45.0", False),
        ("2.54.0", False),
        ("3.0.0", True),
        ("3.1.2", True),
        ("4.0.0", True),
        # Tag-derived version strings the old `int()` parse choked on.
        ("v3.0.0", True),
        ("3.0.0b1", True),
        ("3.0.0.dev1", True),
    ],
)
def test_version_fallback_reads_major_version(
    monkeypatch: pytest.MonkeyPatch, version: str, expected: bool
) -> None:
    """With no readable base class, fall back to the openai major version."""
    monkeypatch.setattr(openai, "DefaultHttpxClient", object, raising=False)
    monkeypatch.setattr(openai, "__version__", version)
    assert _compat._sdk_uses_httpx2() is expected


def test_version_fallback_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Falling back to the version string must be visible in logs.

    A silent fallback is what made the original failure so hard to trace: the
    only symptom was a retried `APIConnectionError` naming no cause.
    """
    monkeypatch.setattr(openai, "DefaultHttpxClient", object, raising=False)
    monkeypatch.setattr(openai, "__version__", "3.0.0")
    with caplog.at_level(logging.WARNING, logger=_compat.__name__):
        assert _compat._sdk_uses_httpx2() is True
    assert "no recognizable" in caplog.text
    assert "3.0.0" in caplog.text


def test_unreadable_sdk_warns_and_falls_back_to_classic(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Neither a base class nor a version: warn loudly, do not raise at import."""
    monkeypatch.setattr(openai, "DefaultHttpxClient", object, raising=False)
    monkeypatch.setattr(openai, "__version__", "not-a-version")
    with caplog.at_level(logging.WARNING, logger=_compat.__name__):
        assert _compat._sdk_uses_httpx2() is False
    assert "unparseable" in caplog.text
    assert "http_client=" in caplog.text


@pytest.mark.parametrize(
    ("version", "expected"),
    [("3.0.0", 3), ("v3.0.0", 3), ("2.45.0", 2), ("10.1.0", 10), ("", None)],
)
def test_sdk_major_version_parsing(
    monkeypatch: pytest.MonkeyPatch, version: str, expected: int | None
) -> None:
    """Leading non-digits are tolerated; an empty version reads as unknown."""
    monkeypatch.setattr(openai, "__version__", version)
    assert _compat._sdk_major_version() == expected
