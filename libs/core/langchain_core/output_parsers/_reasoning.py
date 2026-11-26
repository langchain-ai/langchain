"""Helpers for extracting structured output wrapped in reasoning tags."""

from __future__ import annotations

import json
import re
from json import JSONDecodeError

_OPEN_THINK_RE = re.compile(r"<think(?:\s[^>]*)?>", re.IGNORECASE)
_CLOSE_THINK_RE = re.compile(r"</think\s*>", re.IGNORECASE)
_OPEN_TOOL_CALL_RE = re.compile(r"<tool_call(?:\s[^>]*)?>", re.IGNORECASE)
_CLOSE_TOOL_CALL_RE = re.compile(r"</tool_call\s*>", re.IGNORECASE)


def strip_reasoning_tags(text: str) -> tuple[str, bool]:
    """Remove leading reasoning blocks and `<tool_call>` wrappers from `text`.

    Reasoning models (e.g. DeepSeek-R1) sometimes emit their chain of thought in
    `<think>...</think>` tags before the structured payload, which is occasionally
    wrapped again in `<tool_call>...</tool_call>`. The tags are only stripped when
    they form a prefix or wrapper of the payload, so JSON strings that merely
    contain these literals are left untouched.

    Args:
        text: The raw model output.

    Returns:
        A tuple of the remaining payload and whether it can be parsed. The flag is
        `False` when a `<think>` block is still open, i.e. no payload has started
        yet.
    """
    text = text.lstrip()
    while True:
        match = _OPEN_THINK_RE.match(text)
        if match is None:
            break
        close = _CLOSE_THINK_RE.search(text, match.end())
        if close is None:
            return "", False
        text = text[close.end() :].lstrip()

    match = _OPEN_TOOL_CALL_RE.match(text)
    if match is None:
        return text, True

    payload = text[match.end() :].lstrip()
    try:
        _, end = json.JSONDecoder().raw_decode(payload)
    except JSONDecodeError:
        close = _CLOSE_TOOL_CALL_RE.search(payload)
        if close is None:
            return payload, True
        return payload[: close.start()].strip(), True
    return payload[:end], True
