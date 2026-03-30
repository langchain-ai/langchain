"""Tier-aware tool adaptation utilities.

Provides helpers for adapting tool definitions to different model capability tiers,
reducing token overhead for smaller models without breaking backward compatibility.

!!! warning "Experimental"
    All public APIs in this module are experimental and may change in future releases.
"""

from __future__ import annotations

import re
from collections import Counter
from math import log
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, create_model

if TYPE_CHECKING:
    from langchain_core.tools.base import BaseTool

# ---------------------------------------------------------------------------
# Tier definition
# ---------------------------------------------------------------------------

ModelTier = Literal["small", "medium", "large", "xlarge"]
"""Capability tier for a language model.

- `'small'`:  <=3B parameters   - up to 4 tools, MCQ/condensed format.
- `'medium'`: 4B-14B parameters  - up to 8 tools, condensed format.
- `'large'`:  15B-35B parameters - up to 20 tools, ranked format.
- `'xlarge'`: >35B / frontier   - unlimited tools, full descriptions.
"""

# Size thresholds (in billions of parameters) for tier classification.
_SMALL_MAX_B: float = 3
_MEDIUM_MAX_B: float = 14
_LARGE_MAX_B: float = 35

_TIER_ORDER: list[ModelTier] = ["small", "medium", "large", "xlarge"]

# Per-tier routing config (mirrors yantrikos reference implementation).
_TIER_CONFIG: dict[ModelTier, dict[str, Any]] = {
    "small": {
        "max_tools": 4,
        "description_max_chars": 60,
        "show_parameters": False,
    },
    "medium": {
        "max_tools": 8,
        "description_max_chars": 120,
        "show_parameters": True,
    },
    "large": {
        "max_tools": 20,
        "description_max_chars": 300,
        "show_parameters": True,
    },
    "xlarge": {
        "max_tools": 0,  # unlimited
        "description_max_chars": 0,  # unlimited
        "show_parameters": True,
    },
}

# ---------------------------------------------------------------------------
# Named-model patterns for detect_tier
# (subset of yantrikos MODEL_SIZE_PATTERNS, mapped to our 4-tier scheme)
# ---------------------------------------------------------------------------

_NAMED_MODEL_PATTERNS: dict[str, ModelTier] = {
    # XLarge / frontier
    "gpt-4": "xlarge",
    "gpt-4o": "xlarge",
    "o1": "xlarge",
    "claude": "xlarge",
    "opus": "xlarge",
    "gemini-pro": "xlarge",
    "gemini-ultra": "xlarge",
    "deepseek-chat": "xlarge",
    "command-r-plus": "xlarge",
    "mistral-large": "xlarge",
    # Medium
    "mistral": "medium",
    "mixtral": "medium",
    "gpt-3.5": "medium",
    "llama-2": "medium",
    "llama2": "medium",
    # Small
    "phi-1": "small",
    "phi-2": "small",
    "phi1": "small",
    "phi2": "small",
    "nano": "small",
    "tiny": "small",
}


def detect_tier(model_name: str) -> ModelTier:
    """Detect a model's capability tier from its name.

    Detection order:
    1. Extract parameter count from name (e.g. ``'llama3.2:1b'`` → 1B → ``'small'``).
    2. Match against known named-model patterns.
    3. Apply provider hints (local/GGUF models default to ``'medium'``).
    4. Default to ``'xlarge'`` (safest for unknown frontier models).

    Args:
        model_name: The model name or identifier, optionally with a provider prefix
            separated by ``'/'`` or ``':'``
            (e.g., ``'ollama/llama3.2:1b'``, ``'openai:gpt-4'``).

    Returns:
        The detected tier: ``'small'``, ``'medium'``, ``'large'``, or ``'xlarge'``.

    Examples:
        ```python
        from langchain_core.tools.tier import detect_tier

        detect_tier("llama3.2:1b")      # "small"
        detect_tier("llama3:8b")        # "medium"
        detect_tier("llama3:27b")       # "large"
        detect_tier("gpt-4")            # "xlarge"
        detect_tier("ollama/mistral")   # "medium"
        ```
    """
    if not model_name:
        return "medium"

    name = model_name.lower().strip()

    # Strip provider prefix (e.g., "ollama/", "openai/", "openai:")
    for sep in ("/", ":"):
        if sep in name:
            # Keep only the part after the first separator (the model name)
            name = name.split(sep, 1)[1]
            break

    # 1. Extract parameter count from name (most precise)
    size_match = re.search(r"(?:^|[:\-_.])(\d+(?:\.\d+)?)\s*b(?:\b|_)", name)
    if size_match:
        size_b = float(size_match.group(1))
        if size_b <= _SMALL_MAX_B:
            return "small"
        if size_b <= _MEDIUM_MAX_B:
            return "medium"
        if size_b <= _LARGE_MAX_B:
            return "large"
        return "xlarge"

    # 2. Named model patterns
    for pattern, tier in _NAMED_MODEL_PATTERNS.items():
        if pattern in name:
            return tier

    # 3. Provider/format hints - likely local/smaller models
    _local_hints = ["ollama", "local", "gguf", "q4_", "q8_"]
    if any(hint in model_name.lower() for hint in _local_hints):
        return "medium"

    # 4. Default: xlarge (frontier assumption for unknown model names)
    return "xlarge"


# ---------------------------------------------------------------------------
# get_tier_adapted_tools
# ---------------------------------------------------------------------------


def get_tier_adapted_tools(
    tools: list[BaseTool],
    tier: ModelTier,
) -> list[BaseTool]:
    """Return copies of tools with descriptions and schemas adapted for the given tier.

    Tools without any tier metadata (``tier_descriptions``, ``tier_params``) are
    returned unchanged. This function is fully backward compatible.

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
            tier_descriptions={"small": "Read file", "xlarge": "Read with options"},
            tier_params={"small": ["path"], "xlarge": ["path", "encoding"]},
        )
        def file_read(path: str, encoding: str = "utf-8") -> str:
            \"\"\"Read file with options.\"\"\"

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
# TierRouter
# ---------------------------------------------------------------------------


class TierRouter:
    """Routes tool lists to a model using tier-appropriate presentation strategies.

    For a given model tier the router:

    - Ranks tools by relevance to a query using TF-IDF word-overlap scoring.
    - Applies per-tier limits on the number of tools shown.
    - Adapts tool descriptions and parameter schemas via ``get_tier_adapted_tools``.

    Per-tier defaults (from yantrikos reference benchmarks):

    | Tier    | Max tools | Strategy  |
    |---------|-----------|-----------|
    | small   | 4         | top-K condensed |
    | medium  | 8         | top-K condensed |
    | large   | 20        | ranked full     |
    | xlarge  | unlimited | full pass-through |

    !!! warning "Experimental"
        This class is experimental and may change in future releases.

    Examples:
        ```python
        from langchain_core.tools.tier import TierRouter

        router = TierRouter(tools=my_tools, tier="small")
        relevant = router.route("read a file from disk")
        ```
    """

    def __init__(
        self,
        tools: list[BaseTool],
        tier: ModelTier,
        *,
        max_tools: int | None = None,
    ) -> None:
        """Initialise the router.

        Args:
            tools: The full catalogue of available tools.
            tier: The model capability tier to route for.
            max_tools: Override the default per-tier tool limit.
                Pass ``0`` for unlimited.
        """
        self._tier = tier
        self._config = _TIER_CONFIG[tier]
        self._max_tools = (
            max_tools if max_tools is not None else self._config["max_tools"]
        )
        # Pre-adapt all tools for the tier so description/schema are correct.
        self._tools = get_tier_adapted_tools(tools, tier)
        # Build TF-IDF index over tool names + descriptions.
        self._corpus = [
            f"{t.name} {t.description}"
            for t in self._tools
        ]
        self._idf = _build_idf(self._corpus)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, query: str) -> list[BaseTool]:
        """Return the most relevant tools for *query* capped at the tier limit.

        When the tier is ``'xlarge'`` or ``max_tools`` is 0, all tools are
        returned (full pass-through).

        Args:
            query: The user's intent or task description used for ranking.

        Returns:
            Tier-adapted tools ranked by relevance, capped at the tier limit.
        """
        if self._max_tools == 0:
            return list(self._tools)

        scores = _tfidf_scores(query, self._corpus, self._idf)
        ranked = sorted(range(len(self._tools)), key=lambda i: scores[i], reverse=True)
        top_k = ranked[: self._max_tools]
        # Preserve original order within the selected set.
        top_k_ordered = sorted(top_k)
        return [self._tools[i] for i in top_k_ordered]

    @property
    def tier(self) -> ModelTier:
        """The tier this router is configured for."""
        return self._tier

    @property
    def tools(self) -> list[BaseTool]:
        """All tier-adapted tools in the catalogue."""
        return list(self._tools)


# ---------------------------------------------------------------------------
# to_native_tool
# ---------------------------------------------------------------------------


def to_native_tool(tool: BaseTool, tier: ModelTier) -> dict[str, Any]:
    """Convert a ``BaseTool`` to an OpenAI / Ollama-compatible tool schema dict.

    Applies tier-aware description and parameter adaptation before serialising
    to the standard ``{"type": "function", "function": {...}}`` format used by
    OpenAI and Ollama ``/api/chat`` endpoints.

    Args:
        tool: The LangChain tool to convert.
        tier: The model capability tier to adapt for.

    Returns:
        A dict in OpenAI function-calling format::

            {
                "type": "function",
                "function": {
                    "name": "...",
                    "description": "...",
                    "parameters": { ... },  # JSON Schema object
                },
            }

    Examples:
        ```python
        from langchain_core.tools.tier import to_native_tool

        schema = to_native_tool(file_read_tool, tier="small")
        # Pass schema["function"] directly to model.bind_tools() or Ollama API.
        ```
    """
    description = _resolve_tier_description(tool, tier)

    # Build the parameters JSON schema.
    adapted_schema = _resolve_tier_schema(tool, tier)
    schema_cls = adapted_schema if adapted_schema is not None else tool.args_schema

    if schema_cls is not None:
        try:
            if isinstance(schema_cls, type) and issubclass(schema_cls, BaseModel):
                parameters = schema_cls.model_json_schema()
            elif isinstance(schema_cls, dict):
                parameters = schema_cls
            else:
                parameters = {"type": "object", "properties": {}}
        except Exception:
            parameters = {"type": "object", "properties": {}}
    else:
        parameters = {"type": "object", "properties": {}}

    # Truncate description if tier config specifies a max length.
    max_chars = _TIER_CONFIG[tier]["description_max_chars"]
    if max_chars and len(description) > max_chars:
        description = description[:max_chars].rstrip() + "…"

    # Strip parameters when tier config says not to show them.
    if not _TIER_CONFIG[tier]["show_parameters"]:
        parameters = {"type": "object", "properties": {}}

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": description,
            "parameters": parameters,
        },
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_tier_description(tool: BaseTool, tier: ModelTier) -> str:
    """Return the best available description for *tier*, falling back up the chain."""
    if not tool.tier_descriptions:
        return tool.description

    tier_index = _TIER_ORDER.index(tier)
    for candidate in _TIER_ORDER[tier_index:]:
        if candidate in tool.tier_descriptions:
            return tool.tier_descriptions[candidate]

    return tool.description


def _resolve_tier_schema(tool: BaseTool, tier: ModelTier) -> type | None:
    """Return a filtered Pydantic schema for *tier*, or ``None`` when unchanged."""
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

    schema = tool.args_schema
    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        return None

    original_fields = schema.model_fields
    filtered: dict[str, Any] = {
        k: (v.annotation, v)
        for k, v in original_fields.items()
        if k in params
    }

    if not filtered or set(filtered) == set(original_fields):
        return None

    return create_model(  # type: ignore[call-overload]
        schema.__name__,
        **filtered,
    )


# ---------------------------------------------------------------------------
# Lightweight TF-IDF helpers (no external ML dependencies)
# ---------------------------------------------------------------------------


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_idf(corpus: list[str]) -> dict[str, float]:
    """Build inverse document frequency table for *corpus*."""
    n = len(corpus)
    df: Counter[str] = Counter()
    for doc in corpus:
        df.update(set(_tokenise(doc)))
    return {term: log((n + 1) / (count + 1)) + 1 for term, count in df.items()}


def _tfidf_scores(query: str, corpus: list[str], idf: dict[str, float]) -> list[float]:
    """Return a relevance score per document in *corpus* for *query*."""
    query_terms = Counter(_tokenise(query))
    scores: list[float] = []
    for doc in corpus:
        doc_terms = Counter(_tokenise(doc))
        doc_len = max(sum(doc_terms.values()), 1)
        score = sum(
            (tf / doc_len) * idf.get(term, 0.0) * count
            for term, count in query_terms.items()
            if (tf := doc_terms.get(term, 0)) > 0
        )
        scores.append(score)
    return scores
