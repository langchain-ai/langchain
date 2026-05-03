from langchain_core._api.deprecation import (
    LangChainDeprecationWarning,
    LangChainPendingDeprecationWarning,
    deprecated,
    suppress_langchain_deprecation_warning,
    surface_langchain_deprecation_warnings,
    warn_deprecated,
)

AGENT_DEPRECATION_WARNING = (
    "Use `langchain.agents.create_agent` for new applications. It provides a "
    "more flexible agent factory with middleware support, structured output, "
    "and integration with LangGraph for persistence, streaming, and "
    "human-in-the-loop workflows. Migration guide: "
    "https://docs.langchain.com/oss/python/migrate/langchain-v1"
)


__all__ = [
    "AGENT_DEPRECATION_WARNING",
    "LangChainDeprecationWarning",
    "LangChainPendingDeprecationWarning",
    "deprecated",
    "suppress_langchain_deprecation_warning",
    "surface_langchain_deprecation_warnings",
    "warn_deprecated",
]
