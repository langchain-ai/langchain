"""Code execution tool for Claude models.

[Claude docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool)

This module provides a factory function for creating a code execution tool that allows
Claude to run code in a secure, sandboxed environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anthropic.types.beta import (
        BetaCacheControlEphemeralParam,
        BetaCodeExecutionTool20250825Param,
    )


def code_execution_20250825(
    *,
    cache_control: BetaCacheControlEphemeralParam | None = None,
) -> BetaCodeExecutionTool20250825Param:
    """Create a code execution tool for running code in a sandboxed environment.

    The code execution tool allows Claude to run Bash commands and manipulate
    files in a secure, sandboxed environment. Claude can analyze data, create
    visualizations, perform calculations, and process files.

    When this tool is provided, Claude automatically gains access to:

    - **Bash commands**: Execute shell commands for system operations
    - **File operations**: Create, view, and edit files directly

    Supported models:

    - Claude Opus 4.5
    - Claude Opus 4.1
    - Claude Opus 4
    - Claude Sonnet 4.5
    - Claude Sonnet 4
    - Claude Sonnet 3.7 (deprecated)
    - Claude Haiku 4.5
    - Claude Haiku 3.5

    See the [Claude docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool)
    for more details.

    Args:
        cache_control: Enable prompt caching for this tool definition.

            Use `{'type': 'ephemeral'}` to enable caching.

            Optionally specify a `ttl` of `'5m'` (default) or `'1h'`.

    Returns:
        A code execution tool definition to pass to
            [`bind_tools`][langchain_anthropic.chat_models.ChatAnthropic.bind_tools].

    Example:
        ```python
        from langchain_anthropic import ChatAnthropic, tools

        model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
        model_with_code = model.bind_tools(
            [tools.code_execution_20250825(cache_control={"type": "ephemeral"})],
        )
        response = model_with_code.invoke("Calculate the mean of [1, 2, 3, 4, 5]")
        ```
    """
    if cache_control is not None:
        return {
            "type": "code_execution_20250825",
            "name": "code_execution",
            "cache_control": cache_control,
        }
    return {
        "type": "code_execution_20250825",
        "name": "code_execution",
    }


__all__ = ["code_execution_20250825"]
