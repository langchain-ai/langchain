"""TOON (Token-Oriented Object Notation) encoder for LangChain.

TOON is a human-readable text format for representing structured data.
It provides a more concise and readable alternative to JSON, making it ideal
for logging, debugging prompts, and inspecting agent state.

Note:
    This implementation is inspired by the TOON format originally created by
    Johann Schopplich (https://github.com/jschopplich/toon). This is a clean-room
    reimplementation optimized for LangChain, not a direct port.

Example:
    ```python
    from langchain_core.toon import encode
    from langchain_core.messages import HumanMessage, AIMessage

    conversation = [
        HumanMessage(content="What is TOON?"),
        AIMessage(content="Token-Oriented Object Notation"),
    ]

    print(encode(conversation))
    # Output:
    # [2]:
    #   - type: human
    #     content: What is TOON?
    #   - type: ai
    #     content: Token-Oriented Object Notation
    ```
"""

from __future__ import annotations

from typing import Any, Literal

from .constants import DelimiterType
from .encoder import encode_value
from .normalize import normalize_value

__all__ = ["DelimiterType", "encode"]


def encode(
    input_value: Any,
    *,
    indent: int = 2,
    delimiter: DelimiterType = ",",
    length_marker: Literal["#", False] = False,
) -> str:
    r"""Encode arbitrary Python object to TOON format.

    TOON (Token-Oriented Object Notation) is a human-readable format that
    represents structured data in a concise way. It automatically detects
    patterns in arrays and formats them appropriately:

    - Primitive arrays appear inline: `tags[3]: a,b,c`
    - Arrays of objects with identical keys become tables
    - Nested structures use indentation for clarity
    - Special handling for LangChain types (BaseMessage, Document, etc.)

    Args:
        input_value: Any Python object to encode. Supports primitives, dicts,
            lists, dataclasses, datetime objects, and LangChain-specific types.
        indent: Number of spaces per indentation level. Default is `2`.
        delimiter: Character to use as delimiter in arrays and tables.
            Options are `','`, `'|'`, or `'\t'`. Default is `','`.
        length_marker: If `'#'`, array headers are rendered as `[#N]` instead
            of `[N]`. Useful when length information is critical. Default is `False`.

    Returns:
        TOON-formatted string representation.

    Example:
        ```python
        from langchain_core.toon import encode

        data = {
            "user": {"id": 123, "name": "Alice"},
            "tags": ["python", "ai", "langchain"],
            "scores": [95, 87, 92],
        }

        print(encode(data))
        # Output:
        # user:
        #   id: 123
        #   name: Alice
        # tags[3]: python,ai,langchain
        # scores[3]: 95,87,92
        ```

    Example with custom delimiter:
        ```python
        users = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]

        print(encode(users, delimiter="|"))
        # Output:
        # [2|]{id|name}:
        #   1|Alice
        #   2|Bob
        ```
    """
    normalized = normalize_value(input_value)
    return encode_value(
        normalized, indent=indent, delimiter=delimiter, length_marker=length_marker
    )
