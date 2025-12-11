"""MCP connector toolset for Claude models.

[Claude docs](https://docs.anthropic.com/en/docs/agents-and-tools/mcp-connector)

This module provides a factory function for creating MCP toolset definitions that
connect to remote MCP servers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from anthropic.types.beta import (
        BetaCacheControlEphemeralParam,
        BetaMCPToolConfigParam,
        BetaMCPToolDefaultConfigParam,
        BetaMCPToolsetParam,
    )


class MCPToolConfig(TypedDict, total=False):
    """Configuration for a single tool in the MCP toolset.

    Attributes:
        enabled: Whether this tool is enabled.

            Defaults to `True`.
        defer_loading: If `True`, tool description is not sent to the model
            initially.

            Only loaded when returned via `tool_reference` from tool search.
    """

    enabled: bool
    defer_loading: bool


class MCPDefaultConfig(TypedDict, total=False):
    """Default configuration applied to all tools in the MCP toolset.

    Attributes:
        enabled: Whether tools are enabled by default.

            Defaults to `True`.
        defer_loading: If `True`, tool descriptions are not sent to the model
            initially.

            Only loaded when returned via `tool_reference` from tool search.
    """

    enabled: bool
    defer_loading: bool


def _convert_default_config(
    config: MCPDefaultConfig,
) -> BetaMCPToolDefaultConfigParam:
    """Convert `MCPDefaultConfig` to `BetaMCPToolDefaultConfigParam`."""
    result: BetaMCPToolDefaultConfigParam = {}
    if "enabled" in config:
        result["enabled"] = config["enabled"]
    if "defer_loading" in config:
        result["defer_loading"] = config["defer_loading"]
    return result


def _convert_configs(
    configs: dict[str, MCPToolConfig],
) -> dict[str, BetaMCPToolConfigParam]:
    """Convert tool configs dict to Anthropic format."""
    result: dict[str, BetaMCPToolConfigParam] = {}
    for tool_name, config in configs.items():
        tool_config: BetaMCPToolConfigParam = {}
        if "enabled" in config:
            tool_config["enabled"] = config["enabled"]
        if "defer_loading" in config:
            tool_config["defer_loading"] = config["defer_loading"]
        result[tool_name] = tool_config
    return result


def mcp_toolset(
    *,
    mcp_server_name: str,
    default_config: MCPDefaultConfig | None = None,
    configs: dict[str, MCPToolConfig] | None = None,
    cache_control: BetaCacheControlEphemeralParam | None = None,
) -> BetaMCPToolsetParam:
    """Create an MCP toolset that connects to a remote MCP server.

    The MCP toolset enables Claude to use tools from MCP (Model Context Protocol)
    servers without implementing a separate MCP client. This is useful for
    connecting to remote tool providers.

    See the [Claude docs](https://docs.anthropic.com/en/docs/agents-and-tools/mcp-connector)
    for more details.

    Args:
        mcp_server_name: Must match a server name defined in the `mcp_servers`
            array in call options.
        default_config: Default configuration applied to all tools in this
            toolset.

            Individual tool configs will override these defaults.
        configs: Per-tool configuration overrides.

            Keys are tool names, values are configuration objects.
        cache_control: Enable prompt caching for this tool definition.

            Use `{'type': 'ephemeral'}` to enable caching.

            Optionally specify a `ttl` of `'5m'` (default) or `'1h'`.

    Returns:
        An MCP toolset definition to pass to
            [`bind_tools`][langchain_anthropic.chat_models.ChatAnthropic.bind_tools].

    Example:
        ```python title="Enable all tools from an MCP server"
        from langchain_anthropic import ChatAnthropic, tools

        model = ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            mcp_servers=[
                {
                    "type": "url",
                    "url": "https://example-server.modelcontextprotocol.io/sse",
                    "name": "example-mcp",
                    "authorization_token": "YOUR_TOKEN",
                }
            ],
        )
        response = model.invoke(
            "What tools do you have available?",
            tools=[tools.mcp_toolset(mcp_server_name="example-mcp")],
        )
        ```

        ```python title="Configure tool availability"
        model = ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            mcp_servers=[
                {
                    "type": "url",
                    "url": "https://calendar.example.com/sse",
                    "name": "google-calendar-mcp",
                }
            ],
        )
        model_with_mcp = model.bind_tools(
            [
                tools.mcp_toolset(
                    mcp_server_name="google-calendar-mcp",
                    default_config={"enabled": False},
                    configs={
                        "search_events": {"enabled": True},
                        "create_event": {"enabled": False},
                    },
                    cache_control={"type": "ephemeral"},
                )
            ],
        )
        ```
    """
    if cache_control is not None:
        if configs is not None:
            if default_config is not None:
                return {
                    "type": "mcp_toolset",
                    "mcp_server_name": mcp_server_name,
                    "default_config": _convert_default_config(default_config),
                    "configs": _convert_configs(configs),
                    "cache_control": cache_control,
                }
            return {
                "type": "mcp_toolset",
                "mcp_server_name": mcp_server_name,
                "configs": _convert_configs(configs),
                "cache_control": cache_control,
            }
        if default_config is not None:
            return {
                "type": "mcp_toolset",
                "mcp_server_name": mcp_server_name,
                "default_config": _convert_default_config(default_config),
                "cache_control": cache_control,
            }
        return {
            "type": "mcp_toolset",
            "mcp_server_name": mcp_server_name,
            "cache_control": cache_control,
        }
    if configs is not None:
        if default_config is not None:
            return {
                "type": "mcp_toolset",
                "mcp_server_name": mcp_server_name,
                "default_config": _convert_default_config(default_config),
                "configs": _convert_configs(configs),
            }
        return {
            "type": "mcp_toolset",
            "mcp_server_name": mcp_server_name,
            "configs": _convert_configs(configs),
        }
    if default_config is not None:
        return {
            "type": "mcp_toolset",
            "mcp_server_name": mcp_server_name,
            "default_config": _convert_default_config(default_config),
        }
    return {
        "type": "mcp_toolset",
        "mcp_server_name": mcp_server_name,
    }


__all__ = [
    "MCPDefaultConfig",
    "MCPToolConfig",
    "mcp_toolset",
]
