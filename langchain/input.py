"""Handle chained inputs."""
from typing import Dict, List, Optional

_COLOR_MAPPING = {"blue": 104, "yellow": 103, "red": 101, "green": 102}


def get_color_mapping(
    items: List[str], excluded_colors: Optional[List] = None
) -> Dict[str, str]:
    """Get mapping for items to a support color."""
    colors = list(_COLOR_MAPPING.keys())
    if excluded_colors is not None:
        colors = [c for c in colors if c not in excluded_colors]
    color_mapping = {item: colors[i % len(colors)] for i, item in enumerate(items)}
    return color_mapping


def print_text(text: str, color: Optional[str] = None) -> None:
    """Print text with highlighting and no end characters."""
    if color is None:
        print(text, end="")
    else:
        color_str = _COLOR_MAPPING[color]
        print(f"\x1b[{color_str}m{text}\x1b[0m", end="")


class ChainedInput:
    """Class for working with input that is the result of chains."""

    def __init__(self, text: str, verbose: bool = False):
        """Initialize with verbose flag and initial text."""
        self.verbose = verbose
        if self.verbose:
            print_text(text, None)
        self._input = text

    def add(self, text: str, color: Optional[str] = None) -> None:
        """Add text to input, print if in verbose mode."""
        if self.verbose:
            print_text(text, color)
        self._input += text

    @property
    def input(self) -> str:
        """Return the accumulated input."""
        return self._input
