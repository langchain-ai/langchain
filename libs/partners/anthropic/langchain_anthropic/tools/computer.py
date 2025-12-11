"""Computer use tool for Claude models.

[Claude docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool)

This module provides factory functions for creating computer use tools that enable
Claude to interact with desktop environments.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from anthropic.types.beta import BetaCacheControlEphemeralParam


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
    execute: Callable[..., str | Awaitable[str]],
    *,
    display_width_px: int,
    display_height_px: int,
    display_number: int | None = None,
    enable_zoom: bool | None = None,
    cache_control: BetaCacheControlEphemeralParam | None = None,
) -> StructuredTool:
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
        execute: Callback function for executing computer actions.

            See the [Claude docs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/computer-use-tool#available-actions)
            for the available actions.

            Can be sync or async.
        display_width_px: The width of the display in pixels.
        display_height_px: The height of the display in pixels.
        display_number: Optional X11 display number (e.g., `0`, `1`) for the display.
        enable_zoom: Enable zoom action for detailed screen region inspection.

            When enabled, Claude can zoom into specific screen regions.
        cache_control: Enable prompt caching for this tool definition.

            Use `{'type': 'ephemeral'}` to enable caching.

            Optionally specify a `ttl` of `'5m'` (default) or `'1h'`.

    Returns:
        A `StructuredTool` that can be invoked locally and passed to
            [`bind_tools`][langchain_anthropic.chat_models.ChatAnthropic.bind_tools].

    Example:
        ```python title="Manual tool execution loop"
        import base64
        import io

        from langchain_anthropic import ChatAnthropic, tools
        from langchain.messages import HumanMessage, ToolMessage

        # You'll need a library like pyautogui or similar for actual implementation
        # import pyautogui
        # from PIL import Image


        def execute_computer(
            *,
            action: str,
            coordinate: tuple[int, int] | None = None,
            text: str | None = None,
            key: str | None = None,
            scroll_direction: str | None = None,
            scroll_amount: int | None = None,
            duration: int | None = None,
            region: tuple[int, int, int, int] | None = None,
            **kw,
        ):
            # Placeholder - replace with actual screenshot capture
            # screenshot = pyautogui.screenshot()
            # buffer = io.BytesIO()
            # screenshot.save(buffer, format="PNG")
            # screenshot_b64 = base64.b64encode(buffer.getvalue()).decode()

            if action == "screenshot":
                # Return base64-encoded screenshot
                return "data:image/png;base64,<screenshot_data>"
            elif action == "left_click" and coordinate:
                # pyautogui.click(coordinate[0], coordinate[1])
                return "data:image/png;base64,<screenshot_after_click>"
            elif action == "type" and text:
                # pyautogui.typewrite(text)
                return "data:image/png;base64,<screenshot_after_type>"
            elif action == "key" and key:
                # pyautogui.press(key)
                return "data:image/png;base64,<screenshot_after_key>"
            elif action == "scroll" and scroll_direction:
                # scroll_map = {"up": -1, "down": 1}
                # amount = scroll_map.get(scroll_direction, 0) * (scroll_amount or 1)
                # pyautogui.scroll(amount)
                return "data:image/png;base64,<screenshot_after_scroll>"
            elif action == "zoom" and region:
                # Capture zoomed region
                return "data:image/png;base64,<zoomed_region>"
            elif action == "wait" and duration:
                # time.sleep(duration)
                return "data:image/png;base64,<screenshot_after_wait>"
            return "data:image/png;base64,<screenshot>"


        model = ChatAnthropic(model="claude-opus-4-5-20251101")
        computer_tool = tools.computer_20251124(
            execute=execute_computer,
            display_width_px=1024,
            display_height_px=768,
            enable_zoom=True,
        )
        model_with_computer = model.bind_tools([computer_tool])

        query = HumanMessage(content="Take a screenshot of the desktop")
        response = model_with_computer.invoke([query])

        # Process tool calls in a loop until no more tool calls
        messages = [query, response]

        while response.tool_calls:
            for tool_call in response.tool_calls:
                print(f"Action: {tool_call['args'].get('action')}")
                result = computer_tool.invoke(tool_call["args"])
                tool_msg = ToolMessage(content=result, tool_call_id=tool_call["id"])
                messages.append(tool_msg)

            response = model_with_computer.invoke(messages)
            messages.append(response)

        print(response.content)
        ```

        ```python title="Automatic tool execution"
        from langchain.agents import create_agent
        from langchain_anthropic import ChatAnthropic, tools


        def execute_computer(
            *,
            action: str,
            coordinate: tuple[int, int] | None = None,
            text: str | None = None,
            **kw,
        ):
            # Placeholder implementation - replace with actual screen control
            if action == "screenshot":
                return "data:image/png;base64,<screenshot_data>"
            elif action == "left_click" and coordinate:
                return "data:image/png;base64,<screenshot_after_click>"
            elif action == "type" and text:
                return "data:image/png;base64,<screenshot_after_type>"
            return "data:image/png;base64,<screenshot>"


        agent = create_agent(
            model=ChatAnthropic(model="claude-opus-4-5-20251101"),
            tools=[
                tools.computer_20251124(
                    execute=execute_computer,
                    display_width_px=1024,
                    display_height_px=768,
                    enable_zoom=True,
                )
            ],
        )

        query = {"messages": [{"role": "user", "content": "Take a screenshot"}]}
        result = agent.invoke(query)

        for message in result["messages"]:
            message.pretty_print()
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
    execute: Callable[..., str | Awaitable[str]],
    *,
    display_width_px: int,
    display_height_px: int,
    display_number: int | None = None,
    cache_control: BetaCacheControlEphemeralParam | None = None,
) -> StructuredTool:
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
        execute: Callback function for executing computer actions.

            See the [Claude docs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/computer-use-tool#available-actions)
            for the available actions.

            Can be sync or async.
        display_width_px: The width of the display in pixels.
        display_height_px: The height of the display in pixels.
        display_number: Optional X11 display number (e.g., 0, 1) for the display.
        cache_control: Enable prompt caching for this tool definition.

            Use `{'type': 'ephemeral'}` to enable caching.

            Optionally specify a `ttl` of `'5m'` (default) or `'1h'`.

    Returns:
        A `StructuredTool` that can be invoked locally and passed to
            [`bind_tools`][langchain_anthropic.chat_models.ChatAnthropic.bind_tools].

    Example:
        ```python title="Manual tool execution loop"
        from langchain_anthropic import ChatAnthropic, tools
        from langchain.messages import HumanMessage, ToolMessage

        # You'll need a library like pyautogui or similar for actual implementation
        # import pyautogui


        def execute_computer(
            *,
            action: str,
            coordinate: tuple[int, int] | None = None,
            text: str | None = None,
            key: str | None = None,
            scroll_direction: str | None = None,
            scroll_amount: int | None = None,
            duration: int | None = None,
            **kw,
        ):
            # Placeholder - replace with actual screenshot capture
            if action == "screenshot":
                return "data:image/png;base64,<screenshot_data>"
            elif action == "left_click" and coordinate:
                # pyautogui.click(coordinate[0], coordinate[1])
                return "data:image/png;base64,<screenshot_after_click>"
            elif action == "type" and text:
                # pyautogui.typewrite(text)
                return "data:image/png;base64,<screenshot_after_type>"
            elif action == "key" and key:
                # pyautogui.press(key)
                return "data:image/png;base64,<screenshot_after_key>"
            elif action == "scroll" and scroll_direction:
                return "data:image/png;base64,<screenshot_after_scroll>"
            elif action == "wait" and duration:
                # time.sleep(duration)
                return "data:image/png;base64,<screenshot_after_wait>"
            return "data:image/png;base64,<screenshot>"


        model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
        computer_tool = tools.computer_20250124(
            execute=execute_computer,
            display_width_px=1024,
            display_height_px=768,
        )
        model_with_computer = model.bind_tools([computer_tool])

        query = HumanMessage(content="Take a screenshot of the desktop")
        response = model_with_computer.invoke([query])

        # Process tool calls in a loop until no more tool calls
        messages = [query, response]

        while response.tool_calls:
            for tool_call in response.tool_calls:
                print(f"Action: {tool_call['args'].get('action')}")
                result = computer_tool.invoke(tool_call["args"])
                tool_msg = ToolMessage(content=result, tool_call_id=tool_call["id"])
                messages.append(tool_msg)

            response = model_with_computer.invoke(messages)
            messages.append(response)

        print(response.content)
        ```

        ```python title="Automatic tool execution"
        from langchain.agents import create_agent
        from langchain_anthropic import ChatAnthropic, tools


        def execute_computer(
            *,
            action: str,
            coordinate: tuple[int, int] | None = None,
            text: str | None = None,
            **kw,
        ):
            if action == "screenshot":
                return "data:image/png;base64,<screenshot_data>"
            elif action == "left_click" and coordinate:
                return "data:image/png;base64,<screenshot_after_click>"
            elif action == "type" and text:
                return "data:image/png;base64,<screenshot_after_type>"
            return "data:image/png;base64,<screenshot>"


        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-5-20250929"),
            tools=[
                tools.computer_20250124(
                    execute=execute_computer,
                    display_width_px=1024,
                    display_height_px=768,
                )
            ],
        )

        query = {"messages": [{"role": "user", "content": "Take a screenshot"}]}
        result = agent.invoke(query)

        for message in result["messages"]:
            message.pretty_print()
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
