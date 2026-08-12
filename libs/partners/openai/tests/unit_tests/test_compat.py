"""Tests for the `httpx` / `httpx2` compatibility shim (`_compat`)."""

import importlib
import inspect
import sys
import types
from collections.abc import Iterator

import openai
import pytest

from langchain_openai import _compat
from langchain_openai.chat_models import _client_utils


def _sdk_is_httpx2() -> bool:
    """Whether the installed openai SDK is backed by httpx2 (openai>=3)."""
    return hasattr(openai, "DefaultHttpx2Client")


def test_shim_httpx_matches_installed_openai() -> None:
    """The shim resolves httpx2 iff the installed SDK is httpx2-backed."""
    expected = "httpx2" if _sdk_is_httpx2() else "httpx"
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

    If these were built from a different httpx than the SDK's backing library,
    an httpx transport would be handed to an httpx2 client (or vice versa).
    """
    assert _client_utils.httpx is _compat.httpx
    assert isinstance(_client_utils._DEFAULT_CONNECTION_LIMITS, _compat.httpx.Limits)


@pytest.fixture
def isolated_compat() -> Iterator[types.ModuleType]:
    """Yield `_compat`, then reload it to its true env-derived state.

    The branch-selection tests mutate `openai`/`sys.modules` and reload
    `_compat`; this guarantees the module is restored for later tests even if
    an assertion fails mid-test. Requested *before* `monkeypatch` so its
    teardown (the restoring reload) runs *after* the mutations are undone.
    """
    try:
        yield _compat
    finally:
        importlib.reload(_compat)


def test_selects_classic_httpx_when_sdk_lacks_httpx2_factory(
    isolated_compat: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """openai<3 shape (no `DefaultHttpx2Client`) resolves to classic `httpx`."""
    monkeypatch.delattr(openai, "DefaultHttpx2Client", raising=False)
    reloaded = importlib.reload(isolated_compat)
    assert reloaded.httpx.__name__ == "httpx"


def test_selects_httpx2_when_sdk_exposes_httpx2_factory(
    isolated_compat: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """openai>=3 shape (has `DefaultHttpx2Client`) resolves to `httpx2`.

    Stubs `httpx2` in `sys.modules` so the branch is exercised even in an
    environment where `httpx2` is not installed (i.e. CI on openai<3).
    """
    monkeypatch.setattr(openai, "DefaultHttpx2Client", object(), raising=False)
    if "httpx2" not in sys.modules:
        monkeypatch.setitem(sys.modules, "httpx2", types.ModuleType("httpx2"))
    reloaded = importlib.reload(isolated_compat)
    assert reloaded.httpx.__name__ == "httpx2"
