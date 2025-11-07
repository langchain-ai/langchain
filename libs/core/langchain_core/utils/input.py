"""Handle chained inputs."""

from typing import TextIO

_TEXT_COLOR_MAPPING = {
    "blue": "36;1",
    "yellow": "33;1",
    "pink": "38;5;200",
    "green": "32;1",
    "red": "31;1",
}


def get_color_mapping(
    items: list[str], excluded_colors: list | None = None
) -> dict[str, str]:
    """Get mapping for items to a support color.

    Args:
        items: The items to map to colors.
        excluded_colors: The colors to exclude.

    Returns:
        The mapping of items to colors.
    """
    colors = list(_TEXT_COLOR_MAPPING.keys())
    if excluded_colors is not None:
        colors = [c for c in colors if c not in excluded_colors]
    if not colors:
        msg = "No colors available after applying exclusions."
        raise ValueError(msg)
    return {item: colors[i % len(colors)] for i, item in enumerate(items)}


def get_colored_text(text: str, color: str) -> str:
    """Get colored text.

    Args:
        text: The text to color.
        color: The color to use.

    Returns:
        The colored text.
    """
    color_str = _TEXT_COLOR_MAPPING[color]
    return f"\u001b[{color_str}m\033[1;3m{text}\u001b[0m"


def get_bolded_text(text: str) -> str:
    """Get bolded text.

    Args:
        text: The text to bold.

    Returns:
        The bolded text.
    """
    return f"\033[1m{text}\033[0m"


def print_text(
    text: str, color: str | None = None, end: str = "", file: TextIO | None = None
) -> None:
    """Print text with highlighting and no end characters.

    If a color is provided, the text will be printed in that color.
    If a file is provided, the text will be written to that file.

    Args:
        text: The text to print.
        color: The color to use.
        end: The end character to use.
        file: The file to write to.
    """
    text_to_print = get_colored_text(text, color) if color else text
    print(text_to_print, end=end, file=file)
    if file:
        file.flush()  # ensure all printed content are written to file
