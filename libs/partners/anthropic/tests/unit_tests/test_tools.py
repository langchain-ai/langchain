"""Tests for tool factory functions."""

from typing import Any, cast

from langchain_core.tools import StructuredTool

from langchain_anthropic import tools
from langchain_anthropic.chat_models import _is_builtin_tool


# Dummy execute callbacks for client-executable tools.
# Note: StructuredTool calls functions with **kwargs, but the type hints use TypedDicts
# for documentation purposes. We cast to Any to satisfy mypy in tests.
def _dummy_execute(**kwargs: Any) -> str:
    return "executed"


_dummy_execute_any = cast(Any, _dummy_execute)


class TestWebSearch:
    """Tests for `web_search_20250305` (server-only)."""

    def test_default(self) -> None:
        tool = tools.web_search_20250305()
        assert tool["type"] == "web_search_20250305"
        assert tool["name"] == "web_search"
        assert _is_builtin_tool(tool)

    def test_with_max_uses(self) -> None:
        tool = tools.web_search_20250305(max_uses=5)
        assert tool.get("max_uses") == 5

    def test_with_allowed_domains(self) -> None:
        tool = tools.web_search_20250305(allowed_domains=["example.com", "test.org"])
        assert tool.get("allowed_domains") == ["example.com", "test.org"]

    def test_with_blocked_domains(self) -> None:
        tool = tools.web_search_20250305(blocked_domains=["spam.com"])
        assert tool.get("blocked_domains") == ["spam.com"]

    def test_with_user_location(self) -> None:
        location = {
            "type": "approximate",
            "city": "San Francisco",
            "region": "California",
            "country": "US",
            "timezone": "America/Los_Angeles",
        }
        tool = tools.web_search_20250305(user_location=location)  # type: ignore[arg-type]
        assert tool.get("user_location") == location

    def test_with_cache_control(self) -> None:
        tool = tools.web_search_20250305(cache_control={"type": "ephemeral"})
        assert tool.get("cache_control") == {"type": "ephemeral"}


class TestWebFetch:
    """Tests for `web_fetch_20250910` (server-only)."""

    def test_default(self) -> None:
        tool = tools.web_fetch_20250910()
        assert tool["type"] == "web_fetch_20250910"
        assert tool["name"] == "web_fetch"
        assert _is_builtin_tool(tool)

    def test_with_max_uses(self) -> None:
        tool = tools.web_fetch_20250910(max_uses=10)
        assert tool.get("max_uses") == 10

    def test_with_citations(self) -> None:
        tool = tools.web_fetch_20250910(citations={"enabled": True})
        assert tool.get("citations") == {"enabled": True}

    def test_with_max_content_tokens(self) -> None:
        tool = tools.web_fetch_20250910(max_content_tokens=50000)
        assert tool.get("max_content_tokens") == 50000

    def test_with_domain_filters(self) -> None:
        tool = tools.web_fetch_20250910(
            allowed_domains=["arxiv.org"],
            max_uses=5,
        )
        assert tool.get("allowed_domains") == ["arxiv.org"]
        assert tool.get("max_uses") == 5

    def test_with_cache_control(self) -> None:
        tool = tools.web_fetch_20250910(cache_control={"type": "ephemeral"})
        assert tool.get("cache_control") == {"type": "ephemeral"}

        # Ensure setting default explicitly works
        # (Omission of ttl uses default of 5m server-side)
        tool = tools.web_fetch_20250910(
            cache_control={"type": "ephemeral", "ttl": "5m"}
        )
        assert tool.get("cache_control") == {"type": "ephemeral", "ttl": "5m"}

        tool = tools.web_fetch_20250910(
            cache_control={"type": "ephemeral", "ttl": "1h"}
        )
        assert tool.get("cache_control") == {"type": "ephemeral", "ttl": "1h"}


class TestCodeExecution:
    """Tests for `code_execution_20250825` (server-only)."""

    def test_default(self) -> None:
        tool = tools.code_execution_20250825()
        assert tool["type"] == "code_execution_20250825"
        assert tool["name"] == "code_execution"
        assert _is_builtin_tool(tool)

    def test_with_cache_control(self) -> None:
        tool = tools.code_execution_20250825(cache_control={"type": "ephemeral"})
        assert tool.get("cache_control") == {"type": "ephemeral"}

        # Ensure setting default explicitly works
        # (Omission of ttl uses default of 5m server-side)
        tool = tools.code_execution_20250825(
            cache_control={"type": "ephemeral", "ttl": "5m"}
        )
        assert tool.get("cache_control") == {"type": "ephemeral", "ttl": "5m"}

        tool = tools.code_execution_20250825(
            cache_control={"type": "ephemeral", "ttl": "1h"}
        )
        assert tool.get("cache_control") == {"type": "ephemeral", "ttl": "1h"}


class TestMemory:
    """Tests for `memory_20250818` (client-executable)."""

    def test_returns_structured_tool(self) -> None:
        tool = tools.memory_20250818(_dummy_execute_any)
        assert isinstance(tool, StructuredTool)
        assert tool.name == "memory"

    def test_provider_tool_definition_in_extras(self) -> None:
        tool = tools.memory_20250818(_dummy_execute_any)
        assert tool.extras is not None
        provider_def = tool.extras.get("provider_tool_definition")
        assert provider_def is not None
        assert provider_def["type"] == "memory_20250818"
        assert provider_def["name"] == "memory"

    def test_with_cache_control(self) -> None:
        tool = tools.memory_20250818(
            _dummy_execute_any, cache_control={"type": "ephemeral"}
        )
        assert tool.extras is not None
        provider_def = tool.extras.get("provider_tool_definition")
        assert provider_def is not None
        assert provider_def.get("cache_control") == {"type": "ephemeral"}

    def test_can_invoke(self) -> None:
        call_count = 0

        def my_execute(**kwargs: Any) -> str:
            nonlocal call_count
            call_count += 1
            return f"executed with {kwargs}"

        tool = tools.memory_20250818(cast(Any, my_execute))
        result = tool.invoke({"command": "view", "path": "/test"})
        assert call_count == 1
        assert "command" in result


class TestComputer:
    """Tests for computer use tools (client-executable)."""

    def test_computer_20251124_returns_structured_tool(self) -> None:
        tool = tools.computer_20251124(
            _dummy_execute_any,
            display_width_px=1920,
            display_height_px=1080,
        )
        assert isinstance(tool, StructuredTool)
        assert tool.name == "computer"

    def test_computer_20251124_provider_definition(self) -> None:
        tool = tools.computer_20251124(
            _dummy_execute_any,
            display_width_px=1920,
            display_height_px=1080,
        )
        assert tool.extras is not None
        provider_def = tool.extras.get("provider_tool_definition")
        assert provider_def is not None
        assert provider_def["type"] == "computer_20251124"
        assert provider_def["display_width_px"] == 1920
        assert provider_def["display_height_px"] == 1080

    def test_computer_20251124_with_options(self) -> None:
        tool = tools.computer_20251124(
            _dummy_execute_any,
            display_width_px=1024,
            display_height_px=768,
            display_number=1,
            enable_zoom=True,
        )
        assert tool.extras is not None
        provider_def = tool.extras.get("provider_tool_definition")
        assert provider_def is not None
        assert provider_def.get("display_number") == 1
        assert provider_def.get("enable_zoom") is True

    def test_computer_20250124_returns_structured_tool(self) -> None:
        tool = tools.computer_20250124(
            _dummy_execute_any,
            display_width_px=1920,
            display_height_px=1080,
        )
        assert isinstance(tool, StructuredTool)
        assert tool.name == "computer"

    def test_computer_20250124_provider_definition(self) -> None:
        tool = tools.computer_20250124(
            _dummy_execute_any,
            display_width_px=1920,
            display_height_px=1080,
        )
        assert tool.extras is not None
        provider_def = tool.extras.get("provider_tool_definition")
        assert provider_def is not None
        assert provider_def["type"] == "computer_20250124"

    def test_computer_20250124_with_display_number(self) -> None:
        tool = tools.computer_20250124(
            _dummy_execute_any,
            display_width_px=1920,
            display_height_px=1080,
            display_number=0,
        )
        assert tool.extras is not None
        provider_def = tool.extras.get("provider_tool_definition")
        assert provider_def is not None
        assert provider_def.get("display_number") == 0

    def test_computer_20251124_with_cache_control(self) -> None:
        tool = tools.computer_20251124(
            _dummy_execute_any,
            display_width_px=1920,
            display_height_px=1080,
            cache_control={"type": "ephemeral"},
        )
        assert tool.extras is not None
        provider_def = tool.extras.get("provider_tool_definition")
        assert provider_def is not None
        assert provider_def.get("cache_control") == {"type": "ephemeral"}

    def test_computer_20250124_with_cache_control(self) -> None:
        tool = tools.computer_20250124(
            _dummy_execute_any,
            display_width_px=1920,
            display_height_px=1080,
            cache_control={"type": "ephemeral"},
        )
        assert tool.extras is not None
        provider_def = tool.extras.get("provider_tool_definition")
        assert provider_def is not None
        assert provider_def.get("cache_control") == {"type": "ephemeral"}


class TestTextEditor:
    """Tests for text editor tools (client-executable)."""

    def test_text_editor_20250728_returns_structured_tool(self) -> None:
        tool = tools.text_editor_20250728(_dummy_execute_any)
        assert isinstance(tool, StructuredTool)
        assert tool.name == "str_replace_based_edit_tool"

    def test_text_editor_20250728_provider_definition(self) -> None:
        tool = tools.text_editor_20250728(_dummy_execute_any)
        assert tool.extras is not None
        provider_def = tool.extras.get("provider_tool_definition")
        assert provider_def is not None
        assert provider_def["type"] == "text_editor_20250728"
        assert provider_def["name"] == "str_replace_based_edit_tool"

    def test_text_editor_20250429_returns_structured_tool(self) -> None:
        tool = tools.text_editor_20250429(_dummy_execute_any)
        assert isinstance(tool, StructuredTool)

    def test_text_editor_20250429_provider_definition(self) -> None:
        tool = tools.text_editor_20250429(_dummy_execute_any)
        assert tool.extras is not None
        provider_def = tool.extras.get("provider_tool_definition")
        assert provider_def is not None
        assert provider_def["type"] == "text_editor_20250429"

    def test_text_editor_20250124_returns_structured_tool(self) -> None:
        tool = tools.text_editor_20250124(_dummy_execute_any)
        assert isinstance(tool, StructuredTool)
        assert tool.name == "str_replace_editor"

    def test_text_editor_20250124_provider_definition(self) -> None:
        tool = tools.text_editor_20250124(_dummy_execute_any)
        assert tool.extras is not None
        provider_def = tool.extras.get("provider_tool_definition")
        assert provider_def is not None
        assert provider_def["type"] == "text_editor_20250124"
        assert provider_def["name"] == "str_replace_editor"

    def test_text_editor_20250728_with_cache_control(self) -> None:
        tool = tools.text_editor_20250728(
            _dummy_execute_any, cache_control={"type": "ephemeral"}
        )
        assert tool.extras is not None
        provider_def = tool.extras.get("provider_tool_definition")
        assert provider_def is not None
        assert provider_def.get("cache_control") == {"type": "ephemeral"}

    def test_text_editor_20250728_with_max_characters(self) -> None:
        tool = tools.text_editor_20250728(_dummy_execute_any, max_characters=10000)
        assert tool.extras is not None
        provider_def = tool.extras.get("provider_tool_definition")
        assert provider_def is not None
        assert provider_def.get("max_characters") == 10000


class TestBash:
    """Tests for `bash_20250124` (client-executable)."""

    def test_returns_structured_tool(self) -> None:
        tool = tools.bash_20250124(_dummy_execute_any)
        assert isinstance(tool, StructuredTool)
        assert tool.name == "bash"

    def test_provider_tool_definition_in_extras(self) -> None:
        tool = tools.bash_20250124(_dummy_execute_any)
        assert tool.extras is not None
        provider_def = tool.extras.get("provider_tool_definition")
        assert provider_def is not None
        assert provider_def["type"] == "bash_20250124"
        assert provider_def["name"] == "bash"

    def test_with_cache_control(self) -> None:
        tool = tools.bash_20250124(
            _dummy_execute_any, cache_control={"type": "ephemeral"}
        )
        assert tool.extras is not None
        provider_def = tool.extras.get("provider_tool_definition")
        assert provider_def is not None
        assert provider_def.get("cache_control") == {"type": "ephemeral"}

    def test_can_invoke(self) -> None:
        call_count = 0

        def my_execute(**kwargs: Any) -> str:
            nonlocal call_count
            call_count += 1
            return f"executed with {kwargs}"

        tool = tools.bash_20250124(cast(Any, my_execute))
        result = tool.invoke({"command": "ls -la"})
        assert call_count == 1
        assert "command" in result


class TestToolSearch:
    """Tests for tool search tools (server-only)."""

    def test_tool_search_regex(self) -> None:
        tool = tools.tool_search_regex_20251119()
        assert tool["type"] == "tool_search_tool_regex_20251119"
        assert tool["name"] == "tool_search_tool_regex"
        assert _is_builtin_tool(tool)

    def test_tool_search_bm25(self) -> None:
        tool = tools.tool_search_bm25_20251119()
        assert tool["type"] == "tool_search_tool_bm25_20251119"
        assert tool["name"] == "tool_search_tool_bm25"
        assert _is_builtin_tool(tool)

    def test_tool_search_regex_with_cache_control(self) -> None:
        tool = tools.tool_search_regex_20251119(cache_control={"type": "ephemeral"})
        assert tool.get("cache_control") == {"type": "ephemeral"}

        tool = tools.tool_search_regex_20251119(
            cache_control={"type": "ephemeral", "ttl": "5m"}
        )
        assert tool.get("cache_control") == {"type": "ephemeral", "ttl": "5m"}

        tool = tools.tool_search_regex_20251119(
            cache_control={"type": "ephemeral", "ttl": "1h"}
        )
        assert tool.get("cache_control") == {"type": "ephemeral", "ttl": "1h"}

    def test_tool_search_bm25_with_cache_control(self) -> None:
        tool = tools.tool_search_bm25_20251119(cache_control={"type": "ephemeral"})
        assert tool.get("cache_control") == {"type": "ephemeral"}

        tool = tools.tool_search_bm25_20251119(
            cache_control={"type": "ephemeral", "ttl": "5m"}
        )
        assert tool.get("cache_control") == {"type": "ephemeral", "ttl": "5m"}

        tool = tools.tool_search_bm25_20251119(
            cache_control={"type": "ephemeral", "ttl": "1h"}
        )
        assert tool.get("cache_control") == {"type": "ephemeral", "ttl": "1h"}


class TestMCPToolset:
    """Tests for `mcp_toolset` (server-only)."""

    def test_basic(self) -> None:
        tool = tools.mcp_toolset(mcp_server_name="test-server")
        assert tool["type"] == "mcp_toolset"
        assert tool["mcp_server_name"] == "test-server"
        assert _is_builtin_tool(tool)

    def test_with_default_config(self) -> None:
        tool = tools.mcp_toolset(
            mcp_server_name="test-server",
            default_config={"enabled": False, "defer_loading": True},
        )
        assert tool.get("default_config", {}).get("enabled") is False
        assert tool.get("default_config", {}).get("defer_loading") is True

    def test_with_tool_configs(self) -> None:
        tool = tools.mcp_toolset(
            mcp_server_name="test-server",
            configs={
                "tool_a": {"enabled": True},
                "tool_b": {"enabled": False, "defer_loading": True},
            },
        )
        configs = tool.get("configs")
        assert configs is not None
        assert configs.get("tool_a", {}).get("enabled") is True
        assert configs.get("tool_b", {}).get("enabled") is False
        assert configs.get("tool_b", {}).get("defer_loading") is True

    def test_allowlist_pattern(self) -> None:
        """Test the allowlist pattern: disable all, enable specific."""
        tool = tools.mcp_toolset(
            mcp_server_name="calendar-mcp",
            default_config={"enabled": False},
            configs={
                "search_events": {"enabled": True},
                "create_event": {"enabled": True},
            },
        )
        assert tool.get("default_config", {}).get("enabled") is False
        configs = tool.get("configs")
        assert configs is not None
        assert configs.get("search_events", {}).get("enabled") is True
        assert configs.get("create_event", {}).get("enabled") is True

    def test_denylist_pattern(self) -> None:
        """Test the denylist pattern: enable all, disable specific."""
        tool = tools.mcp_toolset(
            mcp_server_name="calendar-mcp",
            configs={
                "delete_all_events": {"enabled": False},
                "share_publicly": {"enabled": False},
            },
        )
        # No default_config means all enabled by default
        assert "default_config" not in tool
        configs = tool.get("configs")
        assert configs is not None
        assert configs.get("delete_all_events", {}).get("enabled") is False

    def test_with_cache_control(self) -> None:
        tool = tools.mcp_toolset(
            mcp_server_name="test-server",
            cache_control={"type": "ephemeral"},
        )
        assert tool.get("cache_control") == {"type": "ephemeral"}

        # Ensure setting default explicitly works
        # (Omission of ttl uses default of 5m server-side)
        tool = tools.mcp_toolset(
            mcp_server_name="test-server",
            cache_control={"type": "ephemeral", "ttl": "5m"},
        )
        assert tool.get("cache_control") == {"type": "ephemeral", "ttl": "5m"}

        tool = tools.mcp_toolset(
            mcp_server_name="test-server",
            cache_control={"type": "ephemeral", "ttl": "1h"},
        )
        assert tool.get("cache_control") == {"type": "ephemeral", "ttl": "1h"}


class TestTypeExports:
    """Test that types are properly exported."""

    def test_citations_config_type(self) -> None:
        config: tools.CitationsConfig = {"enabled": True}
        assert config["enabled"] is True

    def test_mcp_tool_config_type(self) -> None:
        config: tools.MCPToolConfig = {"enabled": True, "defer_loading": False}
        assert config["enabled"] is True

    def test_mcp_default_config_type(self) -> None:
        config: tools.MCPDefaultConfig = {"enabled": False}
        assert config["enabled"] is False
