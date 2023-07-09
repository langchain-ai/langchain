"""Purely for backwards compatibility."""
from langchain.core.input import get_colored_text, get_bolded_text, get_color_mapping, print_text

__all__ = [
    "get_bolded_text",
    "get_color_mapping",
    "get_colored_text",
    "print_text"
]