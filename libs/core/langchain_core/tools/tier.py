"""Tier-aware tool adaptation utilities.

Provides helpers for adapting tool definitions to different model capability tiers,
reducing token overhead for smaller models without breaking backward compatibility.

!!! warning "Experimental"
    All public APIs in this module are experimental and may change in future releases.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from langchain_core.tools.base import BaseTool

ModelTier = Literal["small", "medium", "large"]
"""Capability tier for a language model.

- `'small'`: Models up to ~3B parameters (e.g., Phi-2, Gemma 2B, Llama 3.2 1B).
- `'medium'`: Models in the 7B–20B range (e.g., Mistral 7B, Llama 3 8B).
- `'large'`: Frontier and large models (e.g., GPT-4, Claude, Gemini Pro).
"""

# Regex patterns used to detect tier from model name strings.
_SMALL_PATTERNS: list[str] = [
    r'\b[123]\.?[0-9]?b\b',   # 1B, 1.5B, 2B, 3B
    r'\bsmall\b',
    r'\bmini\b',
    r'\btiny\b',
    r'\bphi-?[12]\b',          # phi-1, phi-2
    r'\bgemma:?2b\b',
    r'\bqwen[:\-]?1\.?5\b',
]

_MEDIUM_PATTERNS: list[str] = [
    r'\b[78]\.?[0-9]?b\b',    # 7B, 8B
    r'\b1[0-9]\.?[0-9]?b\b',  # 10B–19B
    r'\bmedium\b',
    r'\bmistral\b',
    r'\bllama[:\-]?[23]-?8b\b',
]


def detect_tier(model_name: str) -> ModelTier:
    """Detect a model's capability tier from its name.

    Uses pattern matching on the model name string. Returns `'large'` by default
    when no small or medium pattern matches.

    Args:
        model_name: The name or identifier of the model
            (e.g., `'gpt-4'`, `'llama3.2:1b'`, `'mistral'`).

    Returns:
        The detected tier: `'small'`, `'medium'`, or `'large'`.

    Examples:
        ```python
        from langchain_core.tools.tier import detect_tier

        detect_tier("llama3.2:1b")   # "small"
        detect_tier("mistral")        # "medium"
        detect_tier("gpt-4")          # "large"
        ```
    """
    name_lower = model_name.lower()

    for pattern in _SMALL_PATTERNS:
        if re.search(pattern, name_lower):
            return "small"

    for pattern in _MEDIUM_PATTERNS:
        if re.search(pattern, name_lower):
            return "medium"

    return "large"


def get_tier_adapted_tools(
    tools: list[BaseTool],
    tier: ModelTier,
) -> list[BaseTool]:
    """Return copies of tools with descriptions and schemas adapted for the given tier.

    Tools without any tier metadata (`tier_descriptions`, `tier_params`) are returned
    unchanged. This function is fully backward compatible — tools that predate tier
    support behave identically.

    Args:
        tools: List of tools to adapt.
        tier: The model capability tier to adapt for.

    Returns:
        List of tools adapted for the given tier. Tools without tier metadata are
        included unchanged.

    Examples:
        ```python
        from langchain_core.tools import tool
        from langchain_core.tools.tier import get_tier_adapted_tools

        @tool(
            tier_descriptions={"small": "Read file", "large": "Read file with options"},
            tier_params={"small": ["path"], "large": ["path", "encoding"]},
        )
        def file_read(path: str, encoding: str = "utf-8") -> str:
            \"\"\"Read file with options.\"\"\"
            return open(path, encoding=encoding).read()

        small_tools = get_tier_adapted_tools([file_read], tier="small")
        # small_tools[0].description == "Read file"
        ```
    """
    adapted: list[BaseTool] = []
    for t in tools:
        update: dict[str, Any] = {}

        adapted_desc = _resolve_tier_description(t, tier)
        if adapted_desc != t.description:
            update["description"] = adapted_desc

        adapted_schema = _resolve_tier_schema(t, tier)
        if adapted_schema is not None:
            update["args_schema"] = adapted_schema

        adapted.append(t.model_copy(update=update) if update else t)

    return adapted


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TIER_ORDER: list[ModelTier] = ["small", "medium", "large"]


def _resolve_tier_description(tool: BaseTool, tier: ModelTier) -> str:
    """Return the best available description for `tier`, falling back up the chain."""
    if not tool.tier_descriptions:
        return tool.description

    tier_index = _TIER_ORDER.index(tier)
    # Try the exact tier, then progressively more capable tiers as fallback.
    for candidate in _TIER_ORDER[tier_index:]:
        if candidate in tool.tier_descriptions:
            return tool.tier_descriptions[candidate]

    return tool.description


def _resolve_tier_schema(
    tool: BaseTool, tier: ModelTier
) -> type | None:
    """Return a filtered Pydantic schema for `tier`, or `None` when unchanged."""
    if not tool.tier_params or tool.args_schema is None:
        return None

    tier_index = _TIER_ORDER.index(tier)
    params: list[str] | None = None
    for candidate in _TIER_ORDER[tier_index:]:
        if candidate in tool.tier_params:
            params = tool.tier_params[candidate]
            break

    if params is None:
        return None

    from pydantic import BaseModel, create_model

    if not (isinstance(tool.args_schema, type) and issubclass(tool.args_schema, BaseModel)):
        return None

    original_fields = tool.args_schema.model_fields
    filtered: dict[str, Any] = {
        k: (v.annotation, v)
        for k, v in original_fields.items()
        if k in params
    }

    if not filtered or set(filtered) == set(original_fields):
        # No change in fields — don't create a redundant schema.
        return None

    return create_model(  # type: ignore[call-overload]
        tool.args_schema.__name__,
        **filtered,
    )
