"""Tests for the `httpx` / `httpx2` compatibility shim (`_compat`)."""

import inspect

import openai
import pytest

from langchain_openai import _compat
from langchain_openai.chat_models import _client_utils


def test_module_httpx_matches_helper() -> None:
    """The module-level `httpx` binding agrees with the version check."""
    expected = "httpx2" if _compat._sdk_uses_httpx2() else "httpx"
    assert _compat.httpx.__name__ == expected


def test_only_httpx_is_exported() -> None:
    """The default client classes are intentionally not re-exported."""
    assert _compat.__all__ == ["httpx"]


def test_default_clients_are_subclassable_classes() -> None:
    """`DefaultHttpxClient`/`DefaultAsyncHttpxClient` must be classes, not factories.

    Regression guard: on openai>=3, `DefaultHttpx2Client` is a *factory
    function* (returns an `httpx2.Client`) and cannot be subclassed, while the
    legacy `DefaultHttpxClient` name remains a subclassable class. The wrappers
    below depend on subclassing the latter.
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

    Handing the SDK an `httpx` object when it expects `httpx2` (or vice
    versa) happens to work today because the two libraries are drop-in
    equivalents. Matching them anyway keeps type annotations and `isinstance`
    checks honest.
    """
    assert _client_utils.httpx is _compat.httpx
    assert isinstance(_client_utils._DEFAULT_CONNECTION_LIMITS, _compat.httpx.Limits)


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("2.45.0", False),
        ("2.54.0", False),
        ("3.0.0", True),
        ("3.1.2", True),
        ("4.0.0", True),
        # Tag-derived / pre-release strings a plain `int()` parse choked on.
        ("v3.0.0", True),
        ("3.0.0rc1", True),
        ("3.0.0.dev1", True),
    ],
)
def test_sdk_uses_httpx2_reads_major_version(
    monkeypatch: pytest.MonkeyPatch, version: str, expected: bool
) -> None:
    """Selection keys on the openai major version (httpx2 default landed in 3.0)."""
    monkeypatch.setattr(openai, "__version__", version)
    assert _compat._sdk_uses_httpx2() is expected


def test_httpx2_factory_presence_does_not_flip_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Late openai 2.x exposes a `DefaultHttpx2Client` factory but stays classic.

    Regression guard for the case where a 2.x SDK backports the httpx2 factory:
    the major version (2) must win, so the shim stays on classic `httpx`.
    """
    monkeypatch.setattr(openai, "__version__", "2.54.0")
    monkeypatch.setattr(openai, "DefaultHttpx2Client", object(), raising=False)
    assert _compat._sdk_uses_httpx2() is False


def test_sdk_uses_httpx2_unparseable_version_falls_back_to_classic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unparseable version must not raise at import; fall back to classic."""
    monkeypatch.setattr(openai, "__version__", "not-a-version")
    assert _compat._sdk_uses_httpx2() is False
