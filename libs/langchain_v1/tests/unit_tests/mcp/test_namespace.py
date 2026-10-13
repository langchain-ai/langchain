"""Tests for the `langchain.mcp` namespace itself."""

from __future__ import annotations

import importlib
import warnings

from langchain_core._api import LangChainBetaWarning

import langchain.mcp


def _reimport() -> list[warnings.WarningMessage]:
    """Re-execute `langchain.mcp` and return the warnings it raises.

    The module is already imported by the time any test runs, and a module body
    executes once per process, so the import-time warning has to be provoked
    with a reload rather than observed on a plain import.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(langchain.mcp)
    return caught


def test_importing_the_namespace_warns_that_it_is_in_beta() -> None:
    beta_warnings = [
        warning for warning in _reimport() if issubclass(warning.category, LangChainBetaWarning)
    ]

    assert len(beta_warnings) == 1
    assert "`langchain.mcp` is in beta" in str(beta_warnings[0].message)


def test_the_beta_warning_can_be_silenced() -> None:
    """A warning a caller cannot turn off would be a warning they learn to ignore."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=LangChainBetaWarning)
        importlib.reload(langchain.mcp)

    assert [w for w in caught if issubclass(w.category, LangChainBetaWarning)] == []


def test_the_public_names_survive_the_warning() -> None:
    """The warning must not become the module's only effect."""
    assert sorted(langchain.mcp.__all__) == [
        "MCPAdapter",
        "MCPToolArtifact",
        "as_langchain_tool",
    ]
    for name in langchain.mcp.__all__:
        assert hasattr(langchain.mcp, name)
