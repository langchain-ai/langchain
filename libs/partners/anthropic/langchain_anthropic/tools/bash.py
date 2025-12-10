"""Bash tool for Claude models.

[Claude docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/bash-tool)

This module provides a factory function for creating a bash tool that enables Claude to
execute shell commands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anthropic.types import CacheControlEphemeralParam, ToolBash20250124Param


def bash_20250124(
    *,
    cache_control: CacheControlEphemeralParam | None = None,
) -> ToolBash20250124Param:
    """Create a bash tool for executing shell commands.

    The bash tool enables Claude to execute bash commands in a persistent
    shell session. Claude can run system operations, install packages,
    and interact with the filesystem.

    Supported models:

    - Claude 4 models
    - Sonnet 3.7 (deprecated)

    See the [Claude docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/bash-tool)
    for more details.

    Note:
        This tool is typically used in conjunction with the computer use tool for
        agentic workflows.

    Args:
        cache_control: Enable prompt caching for this tool definition.

            Use `{'type': 'ephemeral'}` to enable caching.

            Optionally specify a `ttl` of `'5m'` (default) or `'1h'`.

    Returns:
        A bash tool definition to pass to
            [`bind_tools`][langchain_anthropic.chat_models.ChatAnthropic.bind_tools].

    Example:
        ```python
        from langchain_anthropic import ChatAnthropic, tools

        model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
        model_with_bash = model.bind_tools(
            [tools.bash_20250124(cache_control={"type": "ephemeral"})],
        )

        response = model_with_bash.invoke("List the files in the current directory")
        ```
    """
    if cache_control is not None:
        return {
            "type": "bash_20250124",
            "name": "bash",
            "cache_control": cache_control,
        }
    return {
        "type": "bash_20250124",
        "name": "bash",
    }


__all__ = ["bash_20250124"]
