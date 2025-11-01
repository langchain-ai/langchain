"""Constants and type definitions for TOON encoding."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Final, Literal

# List markers
LIST_ITEM_MARKER: Final[str] = "-"
LIST_ITEM_PREFIX: Final[str] = "- "

# Structural characters
COMMA: Final = ","
COLON: Final[str] = ":"
SPACE: Final[str] = " "
PIPE: Final = "|"
TAB: Final = "\t"

# Brackets and braces
OPEN_BRACKET: Final[str] = "["
CLOSE_BRACKET: Final[str] = "]"
OPEN_BRACE: Final[str] = "{"
CLOSE_BRACE: Final[str] = "}"

# Literals
NULL_LITERAL: Final[str] = "null"
TRUE_LITERAL: Final[str] = "true"
FALSE_LITERAL: Final[str] = "false"

# Escape characters
BACKSLASH: Final[str] = "\\"
DOUBLE_QUOTE: Final[str] = '"'
NEWLINE: Final[str] = "\n"
CARRIAGE_RETURN: Final[str] = "\r"

# Delimiters
DelimiterType = Literal[",", "|", "\t"]
DEFAULT_DELIMITER: Final[DelimiterType] = COMMA

# Type definitions for JSON-like values
JsonPrimitive = str | int | float | bool | None
JsonObject = MutableMapping[str, "JsonValue"] | Mapping[str, "JsonValue"]
JsonArray = Sequence["JsonValue"]
JsonValue = JsonPrimitive | JsonObject | JsonArray

# Depth tracking for indentation
Depth = int
