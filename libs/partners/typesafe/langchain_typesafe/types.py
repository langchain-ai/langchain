"""Input state accepted by the TypeSafe integration.

Question and answer types come from the TypeSafe SDK and are re-exported by
`langchain_typesafe` rather than redeclared here. This module defines only the one
type the integration adds on top of them: a state type that also accepts LangChain
messages.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

from langchain_core.messages import BaseMessage

_StateValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | BaseMessage
    | Sequence["_StateValue"]
    | dict[str, "_StateValue"]
    | None
)

State: TypeAlias = str | BaseMessage | Sequence[_StateValue] | dict[str, _StateValue]
"""Root state accepted by `TypeSafeClassifier`.

TypeSafe natively accepts a string, JSON object, or JSON array. LangChain
`BaseMessage` objects and message sequences can appear at the root or at any depth
inside objects and arrays. The integration serializes messages as role/content JSON
while preserving surrounding JSON structure.
"""

__all__ = ["State"]
