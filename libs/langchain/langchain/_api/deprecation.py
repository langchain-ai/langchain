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
    "tool-calling, persistence of state, and human-in-the-loop workflows. For "
    "details, refer to the "
    "`LangGraph documentation <https://langchain-ai.github.io/langgraph/>`_"
    " as well as guides for "
    "`Migrating from AgentExecutor <https://python.langchain.com/docs/how_to/migrate_agent/>`_"
    " and LangGraph's "
    "`Pre-built ReAct agent <https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/>`_."
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
