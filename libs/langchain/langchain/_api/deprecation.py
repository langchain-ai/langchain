from langchain_core._api.deprecation import (
    LangChainDeprecationWarning,
    LangChainPendingDeprecationWarning,
    deprecated,
    suppress_langchain_deprecation_warning,
    surface_langchain_deprecation_warnings,
    warn_deprecated,
)

AGENT_DEPRECATION_WARNING = (
    "LangChain agents will continue to be supported, but it is recommended for new "
    "use cases to be built with LangGraph. LangGraph offers a more flexible and "
    "full-featured framework for building agents, including support for "
    "tool-calling, persistence of state, and human-in-the-loop workflows. See "
    "LangGraph documentation for more details: "
    "https://langchain-ai.github.io/langgraph/. Refer here for its pre-built "
    "ReAct agent: "
    "https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/"
)


__all__ = [
    "AGENT_DEPRECATION_WARNING",
    "LangChainDeprecationWarning",
    "LangChainPendingDeprecationWarning",
    "deprecated",
    "suppress_langchain_deprecation_warning",
    "warn_deprecated",
    "surface_langchain_deprecation_warnings",
]
