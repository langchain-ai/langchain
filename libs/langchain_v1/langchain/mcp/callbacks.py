"""Types for callbacks."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from mcp.client.session import ClientRequestContext as MCPRequestContext
from mcp.client.session import ElicitationFnT as MCPElicitationFnT
from mcp.client.session import LoggingFnT as MCPLoggingFnT
from mcp.client.session import ProgressFnT as MCPProgressFnT
from mcp.types import (
    ElicitRequestParams as MCPElicitRequestParams,
)
from mcp.types import (
    ElicitResult as MCPElicitResult,
)
from mcp.types import (
    LoggingMessageNotificationParams as MCPLoggingMessageNotificationParams,
)

from langchain.mcp.elicitation import ElicitationMode, interrupt_for_elicitation

# Type aliases to avoid direct MCP type dependencies
LoggingFnT = MCPLoggingFnT
ProgressFnT = MCPProgressFnT
ElicitationFnT = MCPElicitationFnT
LoggingMessageNotificationParams = MCPLoggingMessageNotificationParams
ElicitRequestParams = MCPElicitRequestParams


@dataclass
class CallbackContext:
    """LangChain MCP client callback context."""

    server_name: str
    tool_name: str | None = None


@runtime_checkable
class LoggingMessageCallback(Protocol):
    """Light wrapper around the mcp.client.session.LoggingFnT.

    Injects callback context as the last argument.
    """

    async def __call__(
        self,
        params: LoggingMessageNotificationParams,
        context: CallbackContext,
    ) -> None:
        """Execute callback on logging message notification."""
        ...


@runtime_checkable
class ProgressCallback(Protocol):
    """Light wrapper around the mcp.client.session.ProgressFnT.

    Injects callback context as the last argument.
    """

    async def __call__(
        self,
        progress: float,
        total: float | None,
        message: str | None,
        context: CallbackContext,
    ) -> None:
        """Execute callback on progress notification."""
        ...


@runtime_checkable
class ElicitationCallback(Protocol):
    """Light wrapper around the mcp.client.session.ElicitationFnT.

    Injects callback context as the last argument.
    """

    async def __call__(
        self,
        mcp_context: MCPRequestContext,
        params: ElicitRequestParams,
        context: CallbackContext,
    ) -> MCPElicitResult:
        """Handle an elicitation request and return the user's response."""
        ...


@dataclass
class _MCPCallbacks:
    """Callbacks compatible with the MCP SDK. For internal use only."""

    logging_callback: LoggingFnT | None = None
    progress_callback: ProgressFnT | None = None
    elicitation_callback: ElicitationFnT | None = None


@dataclass
class Callbacks:
    """Callbacks for the LangChain MCP client."""

    on_logging_message: LoggingMessageCallback | None = None
    on_progress: ProgressCallback | None = None
    on_elicitation: ElicitationCallback | None = None

    def to_mcp_format(
        self,
        *,
        context: CallbackContext,
        elicitation: ElicitationMode | None = None,
    ) -> _MCPCallbacks:
        """Convert LangChain callbacks to MCP SDK callbacks.

        Args:
            context: Provenance to pass to LangChain callbacks.
            elicitation: If `"interrupt"`, surface elicitation through a LangGraph
                interrupt rather than a custom callback.

        Returns:
            MCP SDK-compatible callbacks.

        Raises:
            ValueError: If a custom elicitation callback and interrupt handling are both set.
        """
        if elicitation == "interrupt" and self.on_elicitation is not None:
            msg = 'Set either `on_elicitation` or `elicitation="interrupt"`, not both.'
            raise ValueError(msg)
        if (on_logging_message := self.on_logging_message) is not None:

            async def mcp_logging_callback(
                params: LoggingMessageNotificationParams,
            ) -> None:
                await on_logging_message(params, context)
        else:
            mcp_logging_callback = None

        if (on_progress := self.on_progress) is not None:

            async def mcp_progress_callback(
                progress: float, total: float | None, message: str | None
            ) -> None:
                await on_progress(progress, total, message, context)
        else:
            mcp_progress_callback = None

        if elicitation == "interrupt":

            async def mcp_elicitation_callback(
                mcp_context: MCPRequestContext,  # noqa: ARG001
                params: ElicitRequestParams,
            ) -> MCPElicitResult:
                return interrupt_for_elicitation(params, context)
        elif (on_elicitation := self.on_elicitation) is not None:

            async def mcp_elicitation_callback(
                mcp_context: MCPRequestContext,
                params: ElicitRequestParams,
            ) -> MCPElicitResult:
                return await on_elicitation(mcp_context, params, context)
        else:
            mcp_elicitation_callback = None

        return _MCPCallbacks(
            logging_callback=mcp_logging_callback,
            progress_callback=mcp_progress_callback,
            elicitation_callback=mcp_elicitation_callback,
        )
