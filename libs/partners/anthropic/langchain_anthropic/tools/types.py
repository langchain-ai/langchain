"""Shared types for Anthropic server-side tools.

This module provides `TypedDict` definitions for command and action types
used by various server-side tools.
"""

from __future__ import annotations

from typing import Literal, TypedDict

# =============================================================================
# Memory Tool Command Types
# =============================================================================


class MemoryViewCommand(TypedDict):
    """View the contents of a memory file.

    Attributes:
        command: The command type, must be `'view'`.
        path: The path to the memory file to view.
    """

    command: Literal["view"]
    path: str


class MemoryCreateCommand(TypedDict):
    """Create a new memory file.

    Attributes:
        command: The command type, must be `'create'`.
        path: The path where the memory file will be created.
        content: The initial content of the memory file.
    """

    command: Literal["create"]
    path: str
    content: str


class MemoryStrReplaceCommand(TypedDict):
    """Replace text in a memory file.

    Attributes:
        command: The command type, must be `'str_replace'`.
        path: The path to the memory file.
        old_str: The string to find and replace.
        new_str: The replacement string.
    """

    command: Literal["str_replace"]
    path: str
    old_str: str
    new_str: str


class MemoryInsertCommand(TypedDict):
    """Insert text at a specific line in a memory file.

    Attributes:
        command: The command type, must be `'insert'`.
        path: The path to the memory file.
        insert_line: The line number where text will be inserted.
        new_str: The text to insert.
    """

    command: Literal["insert"]
    path: str
    insert_line: int
    new_str: str


class MemoryDeleteCommand(TypedDict):
    """Delete a memory file.

    Attributes:
        command: The command type, must be `'delete'`.
        path: The path to the memory file to delete.
    """

    command: Literal["delete"]
    path: str


class MemoryRenameCommand(TypedDict):
    """Rename a memory file.

    Attributes:
        command: The command type, must be `'rename'`.
        path: The current path to the memory file.
        new_path: The new path for the memory file.
    """

    command: Literal["rename"]
    path: str
    new_path: str


MemoryCommand = (
    MemoryViewCommand
    | MemoryCreateCommand
    | MemoryStrReplaceCommand
    | MemoryInsertCommand
    | MemoryDeleteCommand
    | MemoryRenameCommand
)
"""Union of all memory tool command types."""


# =============================================================================
# Text Editor Tool Command Types
# =============================================================================


class TextEditorViewCommand(TypedDict, total=False):
    """View the contents of a file.

    Attributes:
        command: The command type, must be `'view'`.
        path: The path to the file to view.
        view_range: Optional tuple of `(start_line, end_line)` to view a range.
    """

    command: Literal["view"]
    path: str
    view_range: tuple[int, int]


class TextEditorStrReplaceCommand(TypedDict):
    """Replace text in a file.

    Attributes:
        command: The command type, must be `'str_replace'`.
        path: The path to the file.
        old_str: The string to find and replace.
        new_str: The replacement string.
    """

    command: Literal["str_replace"]
    path: str
    old_str: str
    new_str: str


class TextEditorCreateCommand(TypedDict):
    """Create a new file.

    Attributes:
        command: The command type, must be `'create'`.
        path: The path where the file will be created.
        file_text: The initial content of the file.
    """

    command: Literal["create"]
    path: str
    file_text: str


class TextEditorInsertCommand(TypedDict):
    """Insert text at a specific line in a file.

    Attributes:
        command: The command type, must be `'insert'`.
        path: The path to the file.
        insert_line: The line number where text will be inserted.
        new_str: The text to insert.
    """

    command: Literal["insert"]
    path: str
    insert_line: int
    new_str: str


TextEditorCommand = (
    TextEditorViewCommand
    | TextEditorStrReplaceCommand
    | TextEditorCreateCommand
    | TextEditorInsertCommand
)
"""Union of all text editor tool command types."""


# =============================================================================
# Bash Tool Command Types
# =============================================================================


class BashExecuteCommand(TypedDict):
    """Execute a bash command.

    Attributes:
        command: The bash command to run.
    """

    command: str


class BashRestartCommand(TypedDict):
    """Restart the bash session.

    Attributes:
        restart: Must be True to restart the session.
    """

    restart: Literal[True]


BashCommand = BashExecuteCommand | BashRestartCommand
"""Union of all bash tool command types."""


# =============================================================================
# Computer Use Action Types
# =============================================================================


class ComputerScreenshotAction(TypedDict):
    """Take a screenshot of the display.

    Attributes:
        action: The action type, must be `'screenshot'`.
    """

    action: Literal["screenshot"]


class ComputerLeftClickAction(TypedDict):
    """Perform a left click at coordinates.

    Attributes:
        action: The action type, must be `'left_click'`.
        coordinate: The `(x, y)` coordinates to click.
    """

    action: Literal["left_click"]
    coordinate: tuple[int, int]


class ComputerRightClickAction(TypedDict):
    """Perform a right click at coordinates.

    Attributes:
        action: The action type, must be `'right_click'`.
        coordinate: The `(x, y)` coordinates to click.
    """

    action: Literal["right_click"]
    coordinate: tuple[int, int]


class ComputerMiddleClickAction(TypedDict):
    """Perform a middle click at coordinates.

    Attributes:
        action: The action type, must be `'middle_click'`.
        coordinate: The `(x, y)` coordinates to click.
    """

    action: Literal["middle_click"]
    coordinate: tuple[int, int]


class ComputerDoubleClickAction(TypedDict):
    """Perform a double click at coordinates.

    Attributes:
        action: The action type, must be `'double_click'`.
        coordinate: The `(x, y)` coordinates to click.
    """

    action: Literal["double_click"]
    coordinate: tuple[int, int]


class ComputerTripleClickAction(TypedDict):
    """Perform a triple click at coordinates.

    Attributes:
        action: The action type, must be `'triple_click'`.
        coordinate: The `(x, y)` coordinates to click.
    """

    action: Literal["triple_click"]
    coordinate: tuple[int, int]


class ComputerLeftClickDragAction(TypedDict):
    """Perform a left click drag from start to end coordinates.

    Attributes:
        action: The action type, must be `'left_click_drag'`.
        start_coordinate: The `(x, y)` starting coordinates.
        end_coordinate: The `(x, y)` ending coordinates.
    """

    action: Literal["left_click_drag"]
    start_coordinate: tuple[int, int]
    end_coordinate: tuple[int, int]


class ComputerLeftMouseDownAction(TypedDict):
    """Press and hold the left mouse button at coordinates.

    Attributes:
        action: The action type, must be `'left_mouse_down'`.
        coordinate: The `(x, y)` coordinates.
    """

    action: Literal["left_mouse_down"]
    coordinate: tuple[int, int]


class ComputerLeftMouseUpAction(TypedDict):
    """Release the left mouse button at coordinates.

    Attributes:
        action: The action type, must be `'left_mouse_up'`.
        coordinate: The `(x, y)` coordinates.
    """

    action: Literal["left_mouse_up"]
    coordinate: tuple[int, int]


class ComputerScrollAction(TypedDict):
    """Scroll at coordinates in a direction.

    Attributes:
        action: The action type, must be `'scroll'`.
        coordinate: The `(x, y)` coordinates to scroll at.
        scroll_direction: The direction to scroll.
        scroll_amount: The amount to scroll.
    """

    action: Literal["scroll"]
    coordinate: tuple[int, int]
    scroll_direction: Literal["up", "down", "left", "right"]
    scroll_amount: int


class ComputerTypeAction(TypedDict):
    """Type text.

    Attributes:
        action: The action type, must be `'type'`.
        text: The text to type.
    """

    action: Literal["type"]
    text: str


class ComputerKeyAction(TypedDict):
    """Press a key or key combination.

    Attributes:
        action: The action type, must be `'key'`.
        key: The key or key combination to press (e.g., `'Return'`, `'ctrl+c'`).
    """

    action: Literal["key"]
    key: str


class ComputerMouseMoveAction(TypedDict):
    """Move the mouse to coordinates.

    Attributes:
        action: The action type, must be `'mouse_move'`.
        coordinate: The `(x, y)` coordinates to move to.
    """

    action: Literal["mouse_move"]
    coordinate: tuple[int, int]


class ComputerHoldKeyAction(TypedDict):
    """Hold a key while performing other actions.

    Attributes:
        action: The action type, must be `'hold_key'`.
        key: The key to hold.
    """

    action: Literal["hold_key"]
    key: str


class ComputerWaitAction(TypedDict, total=False):
    """Wait for a duration.

    Attributes:
        action: The action type, must be `'wait'`.
        duration: Optional duration in seconds to wait.
    """

    action: Literal["wait"]
    duration: int


class ComputerZoomAction(TypedDict):
    """Zoom into a specific region of the screen.

    Only available with `computer_20251124` (Claude Opus 4.5).

    Attributes:
        action: The action type, must be `'zoom'`.
        region: Coordinates `[x1, y1, x2, y2]` defining top-left and
            bottom-right corners of the region.
    """

    action: Literal["zoom"]
    region: tuple[int, int, int, int]


ComputerAction20250124 = (
    ComputerScreenshotAction
    | ComputerLeftClickAction
    | ComputerRightClickAction
    | ComputerMiddleClickAction
    | ComputerDoubleClickAction
    | ComputerTripleClickAction
    | ComputerLeftClickDragAction
    | ComputerLeftMouseDownAction
    | ComputerLeftMouseUpAction
    | ComputerScrollAction
    | ComputerTypeAction
    | ComputerKeyAction
    | ComputerMouseMoveAction
    | ComputerHoldKeyAction
    | ComputerWaitAction
)
"""Union of computer use actions for the `20250124` version."""

ComputerAction20251124 = ComputerAction20250124 | ComputerZoomAction
"""Union of computer use actions for the `20251124` version (includes zoom)."""


__all__ = [
    "BashCommand",
    "BashExecuteCommand",
    "BashRestartCommand",
    "ComputerAction20250124",
    "ComputerAction20251124",
    "ComputerDoubleClickAction",
    "ComputerHoldKeyAction",
    "ComputerKeyAction",
    "ComputerLeftClickAction",
    "ComputerLeftClickDragAction",
    "ComputerLeftMouseDownAction",
    "ComputerLeftMouseUpAction",
    "ComputerMiddleClickAction",
    "ComputerMouseMoveAction",
    "ComputerRightClickAction",
    "ComputerScreenshotAction",
    "ComputerScrollAction",
    "ComputerTripleClickAction",
    "ComputerTypeAction",
    "ComputerWaitAction",
    "ComputerZoomAction",
    "MemoryCommand",
    "MemoryCreateCommand",
    "MemoryDeleteCommand",
    "MemoryInsertCommand",
    "MemoryRenameCommand",
    "MemoryStrReplaceCommand",
    "MemoryViewCommand",
    "TextEditorCommand",
    "TextEditorCreateCommand",
    "TextEditorInsertCommand",
    "TextEditorStrReplaceCommand",
    "TextEditorViewCommand",
]
