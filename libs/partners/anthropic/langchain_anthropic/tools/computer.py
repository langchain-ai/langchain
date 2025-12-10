"""Computer use tool for Claude models.

[Claude docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool)

This module provides factory functions for creating computer use tools that enable
Claude to interact with desktop environments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anthropic.types.beta import (
        BetaCacheControlEphemeralParam,
        BetaToolComputerUse20250124Param,
        BetaToolComputerUse20251124Param,
    )


def computer_20251124(
    *,
    display_width_px: int,
    display_height_px: int,
    display_number: int | None = None,
    enable_zoom: bool | None = None,
    cache_control: BetaCacheControlEphemeralParam | None = None,
) -> BetaToolComputerUse20251124Param:
    """Create a computer use tool for Claude Opus 4.5.

    The computer use tool enables Claude to interact with desktop environments
    through screenshot capture, mouse control, and keyboard input.

    This version (`20251124`) includes the zoom action for detailed screen region
    inspection.

    Supported models: Claude Opus 4.5.

    !!! warning

        Computer use is a beta feature with unique risks. Use a dedicated virtual
        machine or container with minimal privileges. Avoid giving access to
        sensitive data.

    See the [Claude docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool)
    for more details.

    Args:
        display_width_px: The width of the display in pixels.
        display_height_px: The height of the display in pixels.
        display_number: Optional X11 display number (e.g., `0`, `1`) for the display.
        enable_zoom: Enable zoom action for detailed screen region inspection.

            When enabled, Claude can zoom into specific screen regions.
        cache_control: Enable prompt caching for this tool definition.

            Use `{'type': 'ephemeral'}` to enable caching.

            Optionally specify a `ttl` of `'5m'` (default) or `'1h'`.

    Returns:
        A computer use tool definition to pass to
            [`bind_tools`][langchain_anthropic.chat_models.ChatAnthropic.bind_tools].

    Example:
        ```python
        from langchain_anthropic import ChatAnthropic, tools

        model = ChatAnthropic(model="claude-opus-4-5-20251101")
        model_with_computer = model.bind_tools(
            [
                tools.computer_20251124(
                    display_width_px=1024,
                    display_height_px=768,
                    display_number=1,
                    enable_zoom=True,
                    cache_control={"type": "ephemeral"},
                )
            ]
        )
        response = model_with_computer.invoke("Take a screenshot")
        ```
    """
    if cache_control is not None:
        if enable_zoom is not None:
            if display_number is not None:
                return {
                    "type": "computer_20251124",
                    "name": "computer",
                    "display_width_px": display_width_px,
                    "display_height_px": display_height_px,
                    "display_number": display_number,
                    "enable_zoom": enable_zoom,
                    "cache_control": cache_control,
                }
            return {
                "type": "computer_20251124",
                "name": "computer",
                "display_width_px": display_width_px,
                "display_height_px": display_height_px,
                "enable_zoom": enable_zoom,
                "cache_control": cache_control,
            }
        if display_number is not None:
            return {
                "type": "computer_20251124",
                "name": "computer",
                "display_width_px": display_width_px,
                "display_height_px": display_height_px,
                "display_number": display_number,
                "cache_control": cache_control,
            }
        return {
            "type": "computer_20251124",
            "name": "computer",
            "display_width_px": display_width_px,
            "display_height_px": display_height_px,
            "cache_control": cache_control,
        }
    if enable_zoom is not None:
        if display_number is not None:
            return {
                "type": "computer_20251124",
                "name": "computer",
                "display_width_px": display_width_px,
                "display_height_px": display_height_px,
                "display_number": display_number,
                "enable_zoom": enable_zoom,
            }
        return {
            "type": "computer_20251124",
            "name": "computer",
            "display_width_px": display_width_px,
            "display_height_px": display_height_px,
            "enable_zoom": enable_zoom,
        }
    if display_number is not None:
        return {
            "type": "computer_20251124",
            "name": "computer",
            "display_width_px": display_width_px,
            "display_height_px": display_height_px,
            "display_number": display_number,
        }
    return {
        "type": "computer_20251124",
        "name": "computer",
        "display_width_px": display_width_px,
        "display_height_px": display_height_px,
    }


def computer_20250124(
    *,
    display_width_px: int,
    display_height_px: int,
    display_number: int | None = None,
    cache_control: BetaCacheControlEphemeralParam | None = None,
) -> BetaToolComputerUse20250124Param:
    """Create a computer use tool for Claude Sonnet/Opus/Haiku models.

    The computer use tool enables Claude to interact with desktop environments
    through screenshot capture, mouse control, and keyboard input.

    Supported models:

    - Claude Sonnet 4.5
    - Haiku 4.5
    - Opus 4.1
    - Sonnet 4
    - Opus 4
    - Sonnet 3.7 (deprecated)

    !!! warning

        Computer use is a beta feature with unique risks. Use a dedicated virtual
        machine or container with minimal privileges. Avoid giving access to
        sensitive data.

    See the [Claude docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool)
    for more details.

    Args:
        display_width_px: The width of the display in pixels.
        display_height_px: The height of the display in pixels.
        display_number: Optional X11 display number (e.g., 0, 1) for the display.
        cache_control: Enable prompt caching for this tool definition.

            Use `{'type': 'ephemeral'}` to enable caching.

            Optionally specify a `ttl` of `'5m'` (default) or `'1h'`.

    Returns:
        A computer use tool definition to pass to
            [`bind_tools`][langchain_anthropic.chat_models.ChatAnthropic.bind_tools].

    Example:
        ```python
        from langchain_anthropic import ChatAnthropic, tools

        model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
        model_with_computer = model.bind_tools(
            [
                tools.computer_20250124(
                    display_width_px=1024,
                    display_height_px=768,
                    display_number=1,
                    cache_control={"type": "ephemeral"},
                )
            ]
        )
        response = model_with_computer.invoke("Take a screenshot")
        ```
    """
    if cache_control is not None:
        if display_number is not None:
            return {
                "type": "computer_20250124",
                "name": "computer",
                "display_width_px": display_width_px,
                "display_height_px": display_height_px,
                "display_number": display_number,
                "cache_control": cache_control,
            }
        return {
            "type": "computer_20250124",
            "name": "computer",
            "display_width_px": display_width_px,
            "display_height_px": display_height_px,
            "cache_control": cache_control,
        }
    if display_number is not None:
        return {
            "type": "computer_20250124",
            "name": "computer",
            "display_width_px": display_width_px,
            "display_height_px": display_height_px,
            "display_number": display_number,
        }
    return {
        "type": "computer_20250124",
        "name": "computer",
        "display_width_px": display_width_px,
        "display_height_px": display_height_px,
    }


__all__ = [
    "computer_20250124",
    "computer_20251124",
]
