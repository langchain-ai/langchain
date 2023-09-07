"""Common utility functions for LLM APIs."""
import re
from typing import List


def enforce_stop_tokens(text: str, stop: List[str]) -> str:
    """Cut off the text as soon as any stop words occur."""
    return re.split("|".join(stop), text)[0]


def remove_stop_words(text: str, stop: List[str]) -> str:
    """Remove undesired stop words from outputs"""
    pattern = "|".join(stop)
    return re.sub(pattern, "", text)
