import json
from json import JSONDecodeError
from typing import Any, List

from langchain_core.exceptions import OutputParserException


# Adapted from https://github.com/KillianLucas/open-interpreter/blob/5b6080fae1f8c68938a1e4fa8667e3744084ee21/interpreter/utils/parse_partial_json.py
# MIT License
def parse_partial_json(s: str, *, strict: bool = False) -> Any:
    """Parse a JSON string that may be missing closing braces.

    Args:
        s: The JSON string to parse.
        strict: Whether to use strict parsing. Defaults to False.

    Returns:
        The parsed JSON object as a Python dictionary.
    """
    # Attempt to parse the string as-is.
    try:
        return json.loads(s, strict=strict)
    except json.JSONDecodeError:
        pass

    # Initialize variables.
    new_s = ""
    stack = []
    is_inside_string = False
    escaped = False

    # Process each character in the string one at a time.
    for char in s:
        if is_inside_string:
            if char == '"' and not escaped:
                is_inside_string = False
            elif char == "\n" and not escaped:
                char = "\\n"  # Replace the newline character with the escape sequence.
            elif char == "\\":
                escaped = not escaped
            else:
                escaped = False
        else:
            if char == '"':
                is_inside_string = True
                escaped = False
            elif char == "{":
                stack.append("}")
            elif char == "[":
                stack.append("]")
            elif char == "}" or char == "]":
                if stack and stack[-1] == char:
                    stack.pop()
                else:
                    # Mismatched closing character; the input is malformed.
                    return None

        # Append the processed character to the new string.
        new_s += char

    # If we're still inside a string at the end of processing,
    # we need to close the string.
    if is_inside_string:
        new_s += '"'

    # Try to parse mods of string until we succeed or run out of characters.
    while new_s:
        final_s = new_s

        # Close any remaining open structures in the reverse
        # order that they were opened.
        for closing_char in reversed(stack):
            final_s += closing_char

        # Attempt to parse the modified string as JSON.
        try:
            return json.loads(final_s, strict=strict)
        except json.JSONDecodeError:
            # If we still can't parse the string as JSON,
            # try removing the last character
            new_s = new_s[:-1]

    # If we got here, we ran out of characters to remove
    # and still couldn't parse the string as JSON, so return the parse error
    # for the original string.
    return json.loads(s, strict=strict)


def parse_tool_calls(
    raw_tool_calls: List[dict], *, partial: bool = False, strict: bool = False, return_id: bool = False,
) -> List[dict]:
    final_tools = []
    exceptions = []
    for tool_call in raw_tool_calls:
        if "function" not in tool_call:
            continue
        if partial:
            try:
                function_args = parse_partial_json(
                    tool_call["function"]["arguments"], strict=strict
                )
            except JSONDecodeError:
                continue
        else:
            try:
                function_args = json.loads(
                    tool_call["function"]["arguments"], strict=strict
                )
            except JSONDecodeError as e:
                exceptions.append(
                    f"Function {tool_call['function']['name']} arguments:\n\n"
                    f"{tool_call['function']['arguments']}\n\nare not valid JSON. "
                    f"Received JSONDecodeError {e}"
                )
                continue
        parsed = {
            "name": tool_call["function"]["name"] or "",
            "args": function_args or {},
        }
        if return_id:
            parsed["id"] = tool_call["id"]
        final_tools.append(parsed)
    if exceptions:
        raise OutputParserException("\n\n".join(exceptions))
    return final_tools
