"""Tests for the local extensions to ``load_tools``."""

from __future__ import annotations

from typing import Any

import pytest

from langchain_classic.agents.load_tools import (
    _EXTRA_OPTIONAL_TOOLS,
    _get_perplexity_search,
    get_all_tool_names,
    load_tools,
)


def test_perplexity_search_in_registry() -> None:
    """``perplexity_search`` must be a known string-loadable tool name."""
    assert "perplexity_search" in _EXTRA_OPTIONAL_TOOLS


def test_perplexity_search_factory_signature() -> None:
    """The registry entry must be a (factory, extra_keys) tuple."""
    factory, extra_keys = _EXTRA_OPTIONAL_TOOLS["perplexity_search"]
    assert callable(factory)
    assert isinstance(extra_keys, list)


def test_perplexity_search_import_error_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing ``langchain_perplexity`` must produce a friendly install hint."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "langchain_perplexity" or name.startswith("langchain_perplexity."):
            msg = "langchain_perplexity is not installed"
            raise ImportError(msg)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="pip install -U langchain-perplexity"):
        _get_perplexity_search()


def test_load_tools_module_exposes_load_tools() -> None:
    """The local module must expose ``load_tools`` and ``get_all_tool_names``."""
    assert callable(load_tools)
    assert callable(get_all_tool_names)
