"""LangChain MCP adapters for connecting MCP servers with LangChain applications.

Interrupt-driven elicitation has its own types — the interrupt payload, the
answers a run resumes with, and the discriminator to recognize them by. Import
those from `langchain.mcp.elicitation`.

!!! warning "This namespace is in beta"

    `langchain.mcp` is actively being worked on and its API may change. Importing
    from it raises a `LangChainBetaWarning` once per process. Silence it with
    `warnings.filterwarnings("ignore", category=LangChainBetaWarning)`, or scope
    the suppression with `langchain_core._api.suppress_langchain_beta_warning()`.
"""

import warnings

from langchain_core._api import LangChainBetaWarning

from langchain.mcp.adapter import MCPAdapter
from langchain.mcp.tools import MCPToolArtifact, as_langchain_tool

# Warned on import rather than through `@beta`, which annotates a function or
# class and so only fires once something is called. The status belongs to the
# whole namespace, and a caller should learn it when they reach for it —
# including when they import a submodule directly, which runs this module first.
warnings.warn(
    "`langchain.mcp` is in beta. It is actively being worked on, so the API may change.",
    LangChainBetaWarning,
    stacklevel=2,
)

__all__ = [
    "MCPAdapter",
    "MCPToolArtifact",
    "as_langchain_tool",
]
