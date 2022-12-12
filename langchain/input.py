"""Handle chained inputs."""
from typing import Dict, List, Optional

_TEXT_COLOR_MAPPING = {
    "blue": "36;1",
    "yellow": "33;1",
    "pink": "38;5;200",
    "green": "32;1",
}


def get_color_mapping(
    items: List[str], excluded_colors: Optional[List] = None
) -> Dict[str, str]:
    """Get mapping for items to a support color."""
    colors = list(_TEXT_COLOR_MAPPING.keys())
    if excluded_colors is not None:
        colors = [c for c in colors if c not in excluded_colors]
    color_mapping = {item: colors[i % len(colors)] for i, item in enumerate(items)}
    return color_mapping


def print_text(text: str, color: Optional[str] = None, end: str = "") -> None:
    """Print text with highlighting and no end characters."""
    if color is None:
        print(text, end=end)
    else:
        color_str = _TEXT_COLOR_MAPPING[color]
        print(f"\u001b[{color_str}m\033[1;3m{text}\u001b[0m", end=end)
