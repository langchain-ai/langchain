"""Computer use tool for Claude models.

[Claude docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool)

This module provides factory functions for creating computer use tools that enable
Claude to interact with desktop environments.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from langchain_anthropic.tools.types import (
    ComputerAction20250124,
    ComputerAction20251124,
)

if TYPE_CHECKING:
    from anthropic.types.beta import (
        BetaCacheControlEphemeralParam,
        BetaToolComputerUse20250124Param,
        BetaToolComputerUse20251124Param,
    )


_BASE_ACTIONS = [
    "screenshot",
    "left_click",
    "right_click",
    "middle_click",
    "double_click",
    "triple_click",
    "left_click_drag",
    "left_mouse_down",
    "left_mouse_up",
    "scroll",
    "type",
    "key",
    "mouse_move",
    "hold_key",
    "wait",
]


class ComputerInput20250124(BaseModel):
    """Input schema for computer use tool (20250124 version)."""

    action: Literal[
        "screenshot",
        "left_click",
        "right_click",
        "middle_click",
        "double_click",
        "triple_click",
        "left_click_drag",
        "left_mouse_down",
        "left_mouse_up",
        "scroll",
        "type",
        "key",
        "mouse_move",
        "hold_key",
        "wait",
    ] = Field(..., description="The action to perform")

    coordinate: tuple[int, int] | None = Field(
        default=None, description="The (x, y) coordinates for click/mouse actions"
    )

    start_coordinate: tuple[int, int] | None = Field(
        default=None, description="Starting coordinates for drag actions"
    )

    end_coordinate: tuple[int, int] | None = Field(
        default=None, description="Ending coordinates for drag actions"
    )

    text: str | None = Field(default=None, description="Text to type (for type action)")

    key: str | None = Field(
        default=None, description="Key to press (for key/hold_key actions)"
    )

    scroll_direction: Literal["up", "down", "left", "right"] | None = Field(
        default=None, description="Direction to scroll"
    )

    scroll_amount: int | None = Field(default=None, description="Amount to scroll")

    duration: int | None = Field(
        default=None, description="Duration in seconds (for wait action)"
    )


class ComputerInput20251124(BaseModel):
    """Input schema for computer use tool (20251124 version with zoom)."""

    action: Literal[
        "screenshot",
        "left_click",
        "right_click",
        "middle_click",
        "double_click",
        "triple_click",
        "left_click_drag",
        "left_mouse_down",
        "left_mouse_up",
        "scroll",
        "type",
        "key",
        "mouse_move",
        "hold_key",
        "wait",
        "zoom",
    ] = Field(..., description="The action to perform")

    coordinate: tuple[int, int] | None = Field(
        default=None, description="The (x, y) coordinates for click/mouse actions"
    )

    start_coordinate: tuple[int, int] | None = Field(
        default=None, description="Starting coordinates for drag actions"
    )

    end_coordinate: tuple[int, int] | None = Field(
        default=None, description="Ending coordinates for drag actions"
    )

    text: str | None = Field(default=None, description="Text to type (for type action)")

    key: str | None = Field(
        default=None, description="Key to press (for key/hold_key actions)"
    )

    scroll_direction: Literal["up", "down", "left", "right"] | None = Field(
        default=None, description="Direction to scroll"
    )

    scroll_amount: int | None = Field(default=None, description="Amount to scroll")

    duration: int | None = Field(
        default=None, description="Duration in seconds (for wait action)"
    )

    region: tuple[int, int, int, int] | None = Field(
        default=None,
        description="Coordinates [x1, y1, x2, y2] for zoom action",
    )


def computer_20251124(
    *,
    display_width_px: int,
    display_height_px: int,
    execute: Callable[[ComputerAction20251124], str | Awaitable[str]] | None = None,
    display_number: int | None = None,
    enable_zoom: bool | None = None,
    cache_control: BetaCacheControlEphemeralParam | None = None,
) -> BetaToolComputerUse20251124Param | BaseTool:
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
        execute: Optional callback function for client-side execution.

            When provided, returns a `StructuredTool` that can be invoked locally. The
            function receives the action input and should return the result (typically a
            base64-encoded screenshot or action confirmation).

            If not provided, returns a server-side tool definition that Anthropic
            executes.
        display_number: Optional X11 display number (e.g., `0`, `1`) for the display.
        enable_zoom: Enable zoom action for detailed screen region inspection.

            When enabled, Claude can zoom into specific screen regions.
        cache_control: Enable prompt caching for this tool definition.

            Use `{'type': 'ephemeral'}` to enable caching.

            Optionally specify a `ttl` of `'5m'` (default) or `'1h'`.

    Returns:
        If `execute` is provided: A `StructuredTool` that can be invoked locally
            and passed to `bind_tools`.

        If `execute` is not provided: A server-side tool definition dict to pass to
            [`bind_tools`][langchain_anthropic.chat_models.ChatAnthropic.bind_tools].

    Example:
        Server-side execution (Anthropic executes the tool):

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

        Client-side execution (you execute the tool):

        ```python
        from langchain_anthropic import ChatAnthropic, tools


        def execute_computer(action):
            if action["action"] == "screenshot":
                # Capture and return base64-encoded screenshot
                return capture_screenshot()
            elif action["action"] == "left_click":
                click(action["coordinate"][0], action["coordinate"][1])
                return capture_screenshot()
            # Handle other actions...


        model = ChatAnthropic(model="claude-opus-4-5-20251101")
        computer_tool = tools.computer_20251124(
            display_width_px=1024,
            display_height_px=768,
            execute=execute_computer,
        )
        model_with_computer = model.bind_tools([computer_tool])
        ```
    """
    name = "computer"

    # Build the provider tool definition
    provider_tool_def: dict[str, Any] = {
        "type": "computer_20251124",
        "name": name,
        "display_width_px": display_width_px,
        "display_height_px": display_height_px,
    }
    if display_number is not None:
        provider_tool_def["display_number"] = display_number
    if enable_zoom is not None:
        provider_tool_def["enable_zoom"] = enable_zoom
    if cache_control is not None:
        provider_tool_def["cache_control"] = cache_control

    # If no execute callback, return server-side definition
    if execute is None:
        return provider_tool_def  # type: ignore[return-value]

    # Create client-side tool with execute callback
    tool = StructuredTool.from_function(
        func=execute,
        name=name,
        description="Interact with desktop environments through screenshots, "
        "mouse control, and keyboard input. Includes zoom for detailed inspection.",
        args_schema=ComputerInput20251124,
    )

    # Store provider-specific definition in extras
    tool.extras = {
        **(tool.extras or {}),
        "provider_tool_definition": provider_tool_def,
    }

    return tool


def computer_20250124(
    *,
    display_width_px: int,
    display_height_px: int,
    execute: Callable[[ComputerAction20250124], str | Awaitable[str]] | None = None,
    display_number: int | None = None,
    cache_control: BetaCacheControlEphemeralParam | None = None,
) -> BetaToolComputerUse20250124Param | BaseTool:
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
        execute: Optional callback function for client-side execution.

            When provided, returns a `StructuredTool` that can be invoked locally. The
            function receives the action input and should return the result (typically a
            base64-encoded screenshot or action confirmation).

            If not provided, returns a server-side tool definition that Anthropic
            executes.
        display_number: Optional X11 display number (e.g., 0, 1) for the display.
        cache_control: Enable prompt caching for this tool definition.

            Use `{'type': 'ephemeral'}` to enable caching.

            Optionally specify a `ttl` of `'5m'` (default) or `'1h'`.

    Returns:
        If `execute` is provided: A `StructuredTool` that can be invoked locally
            and passed to `bind_tools`.

        If `execute` is not provided: A server-side tool definition dict to pass to
            [`bind_tools`][langchain_anthropic.chat_models.ChatAnthropic.bind_tools].

    Example:
        Server-side execution (Anthropic executes the tool):

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

        Client-side execution (you execute the tool):

        ```python
        from langchain_anthropic import ChatAnthropic, tools


        def execute_computer(action):
            if action["action"] == "screenshot":
                return capture_screenshot()
            elif action["action"] == "left_click":
                click(action["coordinate"][0], action["coordinate"][1])
                return capture_screenshot()
            # Handle other actions...


        model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
        computer_tool = tools.computer_20250124(
            display_width_px=1024,
            display_height_px=768,
            execute=execute_computer,
        )
        model_with_computer = model.bind_tools([computer_tool])
        ```
    """
    name = "computer"

    # Build the provider tool definition
    provider_tool_def: dict[str, Any] = {
        "type": "computer_20250124",
        "name": name,
        "display_width_px": display_width_px,
        "display_height_px": display_height_px,
    }
    if display_number is not None:
        provider_tool_def["display_number"] = display_number
    if cache_control is not None:
        provider_tool_def["cache_control"] = cache_control

    # If no execute callback, return server-side definition
    if execute is None:
        return provider_tool_def  # type: ignore[return-value]

    # Create client-side tool with execute callback
    tool = StructuredTool.from_function(
        func=execute,
        name=name,
        description="Interact with desktop environments through screenshots, "
        "mouse control, and keyboard input.",
        args_schema=ComputerInput20250124,
    )

    # Store provider-specific definition in extras
    tool.extras = {
        **(tool.extras or {}),
        "provider_tool_definition": provider_tool_def,
    }

    return tool


__all__ = [
    "computer_20250124",
    "computer_20251124",
]
