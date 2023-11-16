"""Common utility functions for LLM APIs."""
import re
from typing import List


def enforce_stop_tokens(text: str, stop: List[str]) -> str:
    """Cut off the text as soon as any stop words occur."""

    escaped_stop = [re.escape(word) for word in stop]
    pattern = "|".join(escaped_stop)

    return re.split(pattern, text, maxsplit=1)[0]
