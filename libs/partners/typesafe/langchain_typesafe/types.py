"""Input state accepted by the TypeSafe integration.

Question and answer types come from the TypeSafe SDK. This module defines only the one
type the integration adds on top of them: TypeSafe's own state type widened to accept
LangChain messages.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import typesafe_sdk as ts
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

State: TypeAlias = ts.JSONContent | BaseMessage | Sequence[BaseMessage]
"""State accepted by `TypeSafeClassifier`.

TypeSafe natively accepts a string, JSON object, or JSON array. A `BaseMessage` or a
sequence of messages may also be passed directly and is converted to role/content
JSON. Messages nested inside a larger JSON structure are not converted; use
`convert_to_openai_messages` where that structure is built.
"""


class ClassificationRequest(TypedDict):
    """State and questions for one TypeSafe classification request."""

    state: State
    questions: ts.Questions


__all__ = ["ClassificationRequest", "State"]
