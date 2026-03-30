"""Unit tests for tier-aware tool adaptation (langchain_core.tools.tier)."""

from __future__ import annotations

import pytest

from langchain_core.tools import tool
from langchain_core.tools.tier import (
    ModelTier,
    _resolve_tier_description,
    _resolve_tier_schema,
    detect_tier,
    get_tier_adapted_tools,
)


# ---------------------------------------------------------------------------
# detect_tier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("llama3.2:1b", "small"),
        ("llama3.2:1.5b", "small"),
        ("gemma:2b", "small"),
        ("phi-2", "small"),
        ("phi2", "small"),
        ("qwen:1.5", "small"),
        ("small-model", "small"),
        ("my-tiny-llm", "small"),
        ("my-mini-llm", "small"),
        ("mistral", "medium"),
        ("llama3-8b", "medium"),
        ("llama3:8b", "medium"),
        ("medium-model", "medium"),
        ("llama-13b", "medium"),
        ("gpt-4", "large"),
        ("gpt-3.5-turbo", "large"),
        ("claude-3-opus", "large"),
        ("gemini-pro", "large"),
        ("openai:gpt-4o", "large"),
    ],
)
def test_detect_tier(model_name: str, expected: ModelTier) -> None:
    assert detect_tier(model_name) == expected


# ---------------------------------------------------------------------------
# @tool decorator — tier fields are stored on the tool
# ---------------------------------------------------------------------------


def test_tool_decorator_stores_tier_fields() -> None:
    @tool(
        tier_descriptions={"small": "Short desc", "large": "Long description"},
        tier_params={"small": ["path"], "large": ["path", "encoding"]},
        category="filesystem",
    )
    def file_read(path: str, encoding: str = "utf-8") -> str:
        """Read file contents."""
        return open(path, encoding=encoding).read()  # noqa: SIM115

    assert file_read.tier_descriptions == {"small": "Short desc", "large": "Long description"}
    assert file_read.tier_params == {"small": ["path"], "large": ["path", "encoding"]}
    assert file_read.category == "filesystem"


def test_tool_decorator_tier_fields_default_to_none() -> None:
    @tool
    def simple_tool(x: str) -> str:
        """A simple tool."""
        return x

    assert simple_tool.tier_descriptions is None
    assert simple_tool.tier_params is None
    assert simple_tool.category is None


# ---------------------------------------------------------------------------
# _resolve_tier_description
# ---------------------------------------------------------------------------


def _make_tool_with_descriptions(descriptions: dict[str, str]) -> object:
    @tool(tier_descriptions=descriptions)
    def my_tool(x: str) -> str:
        """Default description."""
        return x

    return my_tool


def test_resolve_tier_description_exact_match() -> None:
    t = _make_tool_with_descriptions({"small": "Short", "large": "Full"})
    assert _resolve_tier_description(t, "small") == "Short"
    assert _resolve_tier_description(t, "large") == "Full"


def test_resolve_tier_description_falls_back_to_more_capable_tier() -> None:
    # Only "large" defined — requesting "small" should fall back to "large".
    t = _make_tool_with_descriptions({"large": "Full"})
    assert _resolve_tier_description(t, "small") == "Full"
    assert _resolve_tier_description(t, "medium") == "Full"


def test_resolve_tier_description_falls_back_to_default() -> None:
    # No matching tier and no fallback — return the base description.
    t = _make_tool_with_descriptions({})
    assert _resolve_tier_description(t, "small") == "Default description."


def test_resolve_tier_description_no_tier_metadata() -> None:
    @tool
    def plain_tool(x: str) -> str:
        """Plain tool."""
        return x

    assert _resolve_tier_description(plain_tool, "small") == "Plain tool."


# ---------------------------------------------------------------------------
# get_tier_adapted_tools
# ---------------------------------------------------------------------------


def test_get_tier_adapted_tools_updates_description() -> None:
    @tool(tier_descriptions={"small": "Short", "large": "Full description"})
    def my_tool(x: str) -> str:
        """Full description."""
        return x

    [adapted] = get_tier_adapted_tools([my_tool], "small")
    assert adapted.description == "Short"
    # Original must not be mutated.
    assert my_tool.description == "Full description."


def test_get_tier_adapted_tools_no_change_when_no_metadata() -> None:
    @tool
    def plain(x: str) -> str:
        """Plain."""
        return x

    [adapted] = get_tier_adapted_tools([plain], "small")
    assert adapted is plain  # Same object — no copy made.


def test_get_tier_adapted_tools_filters_schema_params() -> None:
    @tool(tier_params={"small": ["path"], "large": ["path", "encoding"]})
    def file_read(path: str, encoding: str = "utf-8") -> str:
        """Read a file."""
        return open(path, encoding=encoding).read()  # noqa: SIM115

    [adapted] = get_tier_adapted_tools([file_read], "small")
    schema = adapted.args_schema
    assert schema is not None
    fields = schema.model_fields
    assert "path" in fields
    assert "encoding" not in fields


def test_get_tier_adapted_tools_large_tier_keeps_all_params() -> None:
    @tool(tier_params={"small": ["path"], "large": ["path", "encoding"]})
    def file_read(path: str, encoding: str = "utf-8") -> str:
        """Read a file."""
        return open(path, encoding=encoding).read()  # noqa: SIM115

    [adapted] = get_tier_adapted_tools([file_read], "large")
    # All params present — schema unchanged, same object returned.
    assert adapted is file_read


def test_get_tier_adapted_tools_mixed_list() -> None:
    @tool(tier_descriptions={"small": "Short"})
    def tiered(x: str) -> str:
        """Full."""
        return x

    @tool
    def plain(x: str) -> str:
        """Plain."""
        return x

    adapted_tiered, adapted_plain = get_tier_adapted_tools([tiered, plain], "small")
    assert adapted_tiered.description == "Short"
    assert adapted_plain is plain
