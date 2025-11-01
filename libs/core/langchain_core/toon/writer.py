"""Line writer utility for accumulating formatted lines with indentation."""

from __future__ import annotations

from .constants import Depth


class LineWriter:
    """Accumulates formatted lines with proper indentation.

    This class manages the generation of indented output lines for TOON format.
    It maintains a list of lines and handles indentation based on nesting depth.
    """

    def __init__(self, indent_size: int) -> None:
        """Initialize the line writer.

        Args:
            indent_size: Number of spaces per indentation level.
        """
        self.indent_size = indent_size
        self._lines: list[str] = []
        self._indentation = " " * indent_size

    def push(self, depth: Depth, content: str) -> None:
        """Add a line with the specified indentation depth.

        Args:
            depth: Nesting depth level (0 = no indentation).
            content: Text content to append.
        """
        indent = self._indentation * depth
        self._lines.append(f"{indent}{content}")

    def to_string(self) -> str:
        """Generate the final formatted string.

        Returns:
            All accumulated lines joined with newlines.
        """
        return "\n".join(self._lines)
