"""Utilities for JSON."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from langchain_core.exceptions import OutputParserException

if TYPE_CHECKING:
    from collections.abc import Callable


def _replace_new_line(match: re.Match[str]) -> str:
    """Replace newline characters in a regex match with escaped sequences.

    Args:
        match: Regex match object containing the string to process.

    Returns:
        String with newlines, carriage returns, tabs, and quotes properly escaped.
    """
    value = match.group(2)
    value = re.sub(r"\n", r"\\n", value)
    value = re.sub(r"\r", r"\\r", value)
    value = re.sub(r"\t", r"\\t", value)
    value = re.sub(r'(?<!\\)"', r"\"", value)

    return match.group(1) + value + match.group(3)


def _custom_parser(multiline_string: str | bytes | bytearray) -> str:
    r"""Custom parser for multiline strings.

    The LLM response for `action_input` may be a multiline string containing unescaped
    newlines, tabs or quotes. This function replaces those characters with their escaped
    counterparts. (newlines in JSON must be double-escaped: `\\n`).

    Returns:
        The modified string with escaped newlines, tabs and quotes.
    """
    if isinstance(multiline_string, (bytes, bytearray)):
        multiline_string = multiline_string.decode()

    return re.sub(
        r'("action_input"\:\s*")(.*?)(")',
        _replace_new_line,
        multiline_string,
        flags=re.DOTALL,
    )


# Adapted from https://github.com/KillianLucas/open-interpreter/blob/5b6080fae1f8c68938a1e4fa8667e3744084ee21/interpreter/utils/parse_partial_json.py
# MIT License

# Complete JSON string, or incomplete string through end-of-input (optional
# trailing unpaired backslash). Used so brace tracking can skip string bodies
# without a pure-Python per-character scan.
_STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"|"(?:\\.|[^"\\])*(?:\\)?\Z')


def _ends_with_unescaped_quote(raw: str) -> bool:
    """Return whether `raw` ends with a JSON string-closing quote."""
    if len(raw) < 2 or raw[-1] != '"':
        return False
    n = 0
    i = len(raw) - 2
    while i >= 1 and raw[i] == "\\":
        n += 1
        i -= 1
    return n % 2 == 0


def _unpaired_trailing_backslash(raw: str) -> bool:
    """Return whether `raw` ends with an unpaired backslash escape."""
    n = 0
    i = len(raw) - 1
    while i >= 1 and raw[i] == "\\":
        n += 1
        i -= 1
    return n % 2 == 1


def _repair_string_token(raw: str) -> tuple[str, bool]:
    """Repair a JSON string token and report whether it is still open.

    Escapes literal newlines inside the token (same rules as the previous
    character-walker). Drops an unpaired trailing backslash on incomplete
    strings. Does not append a closing quote for open strings.

    Args:
        raw: A string token matched by `_STRING_RE`, including the opening
            quote and, when complete, the closing quote.

    Returns:
        A tuple of `(repaired_token, is_open)`.
    """
    if "\n" not in raw:
        if _ends_with_unescaped_quote(raw):
            return raw, False
        if _unpaired_trailing_backslash(raw):
            return raw[:-1], True
        return raw, True

    # Literal newlines are rare in streamed `json.dumps` output, but must be
    # escaped the same way as the original character-at-a-time walker.
    out: list[str] = []
    escaped = False
    is_complete = False
    for j, char in enumerate(raw):
        if j == 0:
            out.append(char)
            continue
        if escaped:
            out.append(char)
            escaped = False
        elif char == "\\":
            if j == len(raw) - 1:
                break  # drop unpaired trailing escape
            out.append(char)
            escaped = True
        elif char == '"':
            out.append(char)
            is_complete = True
            break
        elif char == "\n":
            out.append("\\n")
        else:
            out.append(char)
    return "".join(out), not is_complete


def _track_structural_chars(span: str, stack: list[str]) -> bool:
    """Update `stack` for braces/brackets in a non-string span.

    Args:
        span: Substring known to be outside JSON string literals.
        stack: Closing characters still needed for open structures.

    Returns:
        `False` if a mismatched closing character is found, else `True`.
    """
    for char in span:
        if char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in {"}", "]"}:
            if stack and stack[-1] == char:
                stack.pop()
            else:
                return False
    return True


def parse_partial_json(s: str, *, strict: bool = False) -> Any:
    """Parse a JSON string that may be missing closing braces.

    Uses a compiled regex to skip string literals when tracking structure, and
    resumes trimming at `JSONDecodeError.pos` so failed closes do not retry
    `json.loads` once per character. Results match the previous character-walker
    implementation.

    Args:
        s: The JSON string to parse.
        strict: Whether to use strict parsing.

    Returns:
        The parsed JSON object as a Python dictionary.
    """
    # Attempt to parse the string as-is.
    try:
        return json.loads(s, strict=strict)
    except json.JSONDecodeError:
        pass

    parts: list[str] = []
    stack: list[str] = []
    pos = 0
    open_string = False

    for match in _STRING_RE.finditer(s):
        if match.start() > pos:
            span = s[pos : match.start()]
            if not _track_structural_chars(span, stack):
                return None
            parts.append(span)

        repaired, open_string = _repair_string_token(match.group(0))
        parts.append(repaired)
        pos = match.end()

    if pos < len(s):
        span = s[pos:]
        if not _track_structural_chars(span, stack):
            return None
        parts.append(span)

    if open_string:
        parts.append('"')

    # Reverse the stack to get the closing characters.
    stack.reverse()
    closing = "".join(stack)
    new_s = "".join(parts)

    # Try to parse progressively shorter prefixes until one succeeds.
    # `JSONDecodeError.pos` skips characters that cannot start a longer valid
    # prefix, avoiding one `json.loads` call per trimmed character.
    while new_s:
        try:
            return json.loads(new_s + closing, strict=strict)
        except json.JSONDecodeError as e:
            cut = e.pos if e.pos < len(new_s) else len(new_s) - 1
            if cut < 0:
                break
            next_s = new_s[:cut]
            if len(next_s) >= len(new_s):
                next_s = new_s[:-1]
            new_s = next_s

    # If we got here, we ran out of characters to remove
    # and still couldn't parse the string as JSON, so return the parse error
    # for the original string.
    return json.loads(s, strict=strict)


_json_markdown_re = re.compile(r"```(json)?(.*)", re.DOTALL)


def parse_json_markdown(
    json_string: str, *, parser: Callable[[str], Any] = parse_partial_json
) -> Any:
    """Parse a JSON string from a Markdown string.

    Args:
        json_string: The Markdown string.
        parser: The parser to use.

    Returns:
        The parsed JSON object as a Python dictionary.
    """
    try:
        return _parse_json(json_string, parser=parser)
    except json.JSONDecodeError:
        # Try to find JSON string within triple backticks
        match = _json_markdown_re.search(json_string)

        # If no match found, assume the entire string is a JSON string
        # Else, use the content within the backticks
        json_str = json_string if match is None else match.group(2)
    return _parse_json(json_str, parser=parser)


_json_strip_chars = " \n\r\t`"


def _parse_json(
    json_str: str, *, parser: Callable[[str], Any] = parse_partial_json
) -> Any:
    """Parse a JSON string, handling special characters and whitespace.

    Strips whitespace, newlines, and backticks from the start and end of the string,
    then processes special characters before parsing.

    Args:
        json_str: The JSON string to parse.
        parser: Optional custom parser function.

    Returns:
        Parsed JSON object.
    """
    # Strip whitespace,newlines,backtick from the start and end
    json_str = json_str.strip(_json_strip_chars)

    # handle newlines and other special characters inside the returned value
    json_str = _custom_parser(json_str)

    # Parse the JSON string into a Python dictionary
    return parser(json_str)


def parse_and_check_json_markdown(
    text: str, expected_keys: list[str]
) -> dict[str, Any]:
    """Parse and check a JSON string from a Markdown string.

    Checks that it contains the expected keys.

    Args:
        text: The Markdown string.
        expected_keys: The expected keys in the JSON string.

    Returns:
        The parsed JSON object as a Python dictionary.

    Raises:
        OutputParserException: If the JSON string is invalid or does not contain
            the expected keys.
    """
    try:
        json_obj = parse_json_markdown(text)
    except json.JSONDecodeError as e:
        msg = f"Got invalid JSON object. Error: {e}"
        raise OutputParserException(msg) from e
    if not isinstance(json_obj, dict):
        error_message = (
            f"Expected JSON object (dict), but got: {type(json_obj).__name__}. "
        )
        raise OutputParserException(error_message, llm_output=text)

    for key in expected_keys:
        if key not in json_obj:
            msg = (
                f"Got invalid return object. Expected key `{key}` "
                f"to be present, but got {json_obj}"
            )
            raise OutputParserException(msg)
    return json_obj
