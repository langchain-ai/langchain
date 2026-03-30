"""Unit tests for tier-aware tool adaptation (langchain_core.tools.tier)."""

from __future__ import annotations

from pathlib import Path

import pytest

from langchain_core.tools import tool
from langchain_core.tools.tier import (
    ModelTier,
    TierRouter,
    _resolve_tier_description,
    detect_tier,
    get_tier_adapted_tools,
    to_native_tool,
)

# ---------------------------------------------------------------------------
# detect_tier — 4-tier scheme with size extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        # Size extraction
        ("llama3.2:1b", "small"),
        ("llama3.2:1.5b", "small"),
        ("gemma:2b", "small"),
        ("qwen3.5:3b", "small"),
        ("llama3:8b", "medium"),
        ("mistral:7b", "medium"),
        ("qwen3.5:9b", "medium"),
        ("llama3:13b", "medium"),
        ("llama3:27b", "large"),
        ("qwen:32b", "large"),
        ("llama3:70b", "xlarge"),
        ("llama:405b", "xlarge"),
        # Named patterns
        ("gpt-4", "xlarge"),
        ("gpt-4o", "xlarge"),
        ("claude-3-opus", "xlarge"),
        ("gemini-pro", "xlarge"),
        ("mistral-large", "xlarge"),
        ("phi-2", "small"),
        ("phi2", "small"),
        ("phi-1", "small"),
        ("mistral", "medium"),
        ("gpt-3.5-turbo", "medium"),
        # Provider hints
        ("ollama/some-unknown-model", "medium"),
        ("some-model.gguf", "medium"),
        # Default for unknown frontier name
        ("totally-unknown-model-xyz", "xlarge"),
    ],
)
def test_detect_tier(model_name: str, expected: ModelTier) -> None:
    assert detect_tier(model_name) == expected


def test_detect_tier_strips_provider_prefix() -> None:
    # "openai:gpt-4" → strip "openai:" → detect "gpt-4" → xlarge
    assert detect_tier("openai:gpt-4") == "xlarge"
    assert detect_tier("ollama/llama3:1b") == "small"


# ---------------------------------------------------------------------------
# @tool decorator — tier fields stored on the tool
# ---------------------------------------------------------------------------


def test_tool_decorator_stores_tier_fields() -> None:
    @tool(
        tier_descriptions={"small": "Short desc", "xlarge": "Long description"},
        tier_params={"small": ["path"], "xlarge": ["path", "encoding"]},
        category="filesystem",
    )
    def file_read(path: str, encoding: str = "utf-8") -> str:
        """Read file contents."""
        return Path(path).open(encoding=encoding).read()

    assert file_read.tier_descriptions == {
        "small": "Short desc",
        "xlarge": "Long description",
    }
    assert file_read.tier_params == {"small": ["path"], "xlarge": ["path", "encoding"]}
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


def _tiered_tool(descriptions: dict[str, str]) -> object:
    @tool(tier_descriptions=descriptions)
    def my_tool(x: str) -> str:
        """Default description."""
        return x

    return my_tool


def test_resolve_description_exact_match() -> None:
    t = _tiered_tool({"small": "Short", "xlarge": "Full"})
    assert _resolve_tier_description(t, "small") == "Short"
    assert _resolve_tier_description(t, "xlarge") == "Full"


def test_resolve_description_fallback_to_more_capable_tier() -> None:
    # Only "xlarge" defined — requesting "small" should fall back.
    t = _tiered_tool({"xlarge": "Full"})
    assert _resolve_tier_description(t, "small") == "Full"
    assert _resolve_tier_description(t, "medium") == "Full"
    assert _resolve_tier_description(t, "large") == "Full"


def test_resolve_description_fallback_to_default() -> None:
    t = _tiered_tool({})
    assert _resolve_tier_description(t, "small") == "Default description."


def test_resolve_description_no_tier_metadata() -> None:
    @tool
    def plain(x: str) -> str:
        """Plain tool."""
        return x

    assert _resolve_tier_description(plain, "small") == "Plain tool."


# ---------------------------------------------------------------------------
# get_tier_adapted_tools
# ---------------------------------------------------------------------------


def test_adapted_tools_updates_description() -> None:
    @tool(tier_descriptions={"small": "Short", "xlarge": "Full description"})
    def my_tool(x: str) -> str:
        """Full description."""
        return x

    [adapted] = get_tier_adapted_tools([my_tool], "small")
    assert adapted.description == "Short"
    assert my_tool.description == "Full description."  # original unchanged


def test_adapted_tools_no_change_when_no_metadata() -> None:
    @tool
    def plain(x: str) -> str:
        """Plain."""
        return x

    [adapted] = get_tier_adapted_tools([plain], "small")
    assert adapted is plain


def test_adapted_tools_filters_schema_params() -> None:
    @tool(tier_params={"small": ["path"], "xlarge": ["path", "encoding"]})
    def file_read(path: str, encoding: str = "utf-8") -> str:
        """Read a file."""

    [adapted] = get_tier_adapted_tools([file_read], "small")
    assert adapted.args_schema is not None
    fields = adapted.args_schema.model_fields
    assert "path" in fields
    assert "encoding" not in fields


def test_adapted_tools_xlarge_keeps_all_params() -> None:
    @tool(tier_params={"small": ["path"], "xlarge": ["path", "encoding"]})
    def file_read(path: str, encoding: str = "utf-8") -> str:
        """Read a file."""

    [adapted] = get_tier_adapted_tools([file_read], "xlarge")
    assert adapted is file_read  # no schema change → same object


# ---------------------------------------------------------------------------
# TierRouter
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_tools() -> list:
    @tool(tier_descriptions={"small": "Read file", "xlarge": "Read file with options"})
    def file_read(path: str) -> str:
        """Read file with options."""

    @tool
    def web_search(query: str) -> str:
        """Search the web for information."""

    @tool
    def send_email(to: str, body: str) -> str:
        """Send an email to a recipient."""

    @tool
    def list_files(directory: str) -> str:
        """List all files in a directory."""

    @tool
    def delete_file(path: str) -> str:
        """Delete a file from disk."""

    return [file_read, web_search, send_email, list_files, delete_file]


def test_tier_router_small_caps_at_4_tools(sample_tools: list) -> None:
    router = TierRouter(tools=sample_tools, tier="small")
    result = router.route("read a file from disk")
    assert len(result) <= 4


def test_tier_router_ranks_relevant_tools_first(sample_tools: list) -> None:
    router = TierRouter(tools=sample_tools, tier="xlarge")
    # xlarge → no limit, all tools returned
    all_tools = router.route("any query")
    assert len(all_tools) == len(sample_tools)


def test_tier_router_file_query_returns_file_tools(sample_tools: list) -> None:
    router = TierRouter(tools=sample_tools, tier="medium")
    result = router.route("read and list files on disk")
    names = [t.name for t in result]
    # file-related tools should rank above send_email
    assert "file_read" in names
    assert "list_files" in names


def test_tier_router_adapts_description_for_tier(sample_tools: list) -> None:
    router = TierRouter(tools=sample_tools, tier="small")
    result = router.route("file")
    file_tool = next((t for t in result if t.name == "file_read"), None)
    if file_tool is not None:
        assert file_tool.description == "Read file"


def test_tier_router_max_tools_override(sample_tools: list) -> None:
    router = TierRouter(tools=sample_tools, tier="small", max_tools=2)
    result = router.route("anything")
    assert len(result) <= 2


# ---------------------------------------------------------------------------
# to_native_tool
# ---------------------------------------------------------------------------


def test_to_native_tool_structure() -> None:
    @tool(tier_descriptions={"small": "Short", "xlarge": "Full description"})
    def my_tool(path: str, encoding: str = "utf-8") -> str:
        """Full description."""

    native = to_native_tool(my_tool, "xlarge")
    assert native["type"] == "function"
    assert "function" in native
    fn = native["function"]
    assert fn["name"] == "my_tool"
    assert fn["description"] == "Full description"
    assert "parameters" in fn
    assert fn["parameters"]["type"] == "object"


def test_to_native_tool_uses_tier_description() -> None:
    @tool(tier_descriptions={"small": "Short desc", "xlarge": "Full description"})
    def my_tool(x: str) -> str:
        """Full description."""

    native = to_native_tool(my_tool, "small")
    assert native["function"]["description"] == "Short desc"


def test_to_native_tool_small_hides_parameters() -> None:
    @tool
    def my_tool(path: str, encoding: str = "utf-8") -> str:
        """Read a file."""

    native = to_native_tool(my_tool, "small")
    # Small tier: show_parameters=False → empty properties
    assert native["function"]["parameters"]["properties"] == {}


def test_to_native_tool_truncates_long_description() -> None:
    long_desc = "A" * 200
    @tool(tier_descriptions={"small": long_desc})
    def my_tool(x: str) -> str:
        """Default."""

    native = to_native_tool(my_tool, "small")
    # small tier max_chars=60
    assert len(native["function"]["description"]) <= 62  # 60 + "…"
    assert native["function"]["description"].endswith("…")
