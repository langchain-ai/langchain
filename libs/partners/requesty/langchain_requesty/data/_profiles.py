"""Model profiles for `langchain-requesty`.

Requesty is an OpenAI-compatible LLM gateway that routes to hundreds of models
from many providers. Because the catalog is large and changes frequently, no
static per-model profiles are shipped here; ``ChatRequesty`` falls back to the
default profile behavior of ``BaseChatOpenAI``.
"""

from typing import Any

_PROFILES: dict[str, dict[str, Any]] = {}
