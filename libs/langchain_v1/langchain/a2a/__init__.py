"""A2A (Agent2Agent) protocol integration for LangChain and LangGraph.

Provides server utilities to expose LangGraph/LangChain agents over A2A
(`to_a2a`, `LangGraphAgentExecutor`) and client primitives to call remote
A2A agents from agents and graphs (`A2ATool`, `A2ARemoteGraph`).
"""

from __future__ import annotations

import warnings

from langchain_core._api import LangChainBetaWarning

warnings.warn(
    "The `langchain.a2a` module is in beta and its API may change in future releases.",
    LangChainBetaWarning,
    stacklevel=2,
)

from langchain.a2a._conversion import A2AExecutionError  # noqa: E402
from langchain.a2a.client import A2ARemoteGraph, A2ATool  # noqa: E402
from langchain.a2a.server import LangGraphAgentExecutor, to_a2a  # noqa: E402

__all__ = [
    "A2AExecutionError",
    "A2ARemoteGraph",
    "A2ATool",
    "LangGraphAgentExecutor",
    "to_a2a",
]
