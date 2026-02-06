"""Trust-gated tools for LangChain.

This module provides tools that require trust verification before execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union

from langchain_core.tools import BaseTool, StructuredTool

from langchain_agentmesh.identity import CMVKIdentity
from langchain_agentmesh.trust import (
    TrustHandshake,
    TrustPolicy,
    TrustVerificationResult,
    TrustedAgentCard,
)


@dataclass
class ToolInvocationRecord:
    """Record of a tool invocation for auditing."""

    tool_name: str
    invoker_did: Optional[str]
    timestamp: datetime
    verified: bool
    trust_score: float
    input_summary: str
    result_summary: str
    warnings: List[str] = field(default_factory=list)


class TrustGatedTool:
    """A wrapper that adds trust verification to any LangChain tool.

    This wrapper ensures that only verified agents with sufficient trust
    and required capabilities can execute the wrapped tool.
    """

    def __init__(
        self,
        tool: Union[BaseTool, Callable],
        required_capabilities: Optional[List[str]] = None,
        min_trust_score: float = 0.7,
        description_suffix: str = "",
    ):
        """Initialize trust-gated tool.

        Args:
            tool: The underlying LangChain tool or callable
            required_capabilities: Capabilities required to use this tool
            min_trust_score: Minimum trust score required
            description_suffix: Additional text to add to tool description
        """
        self.tool = tool
        self.required_capabilities = required_capabilities or []
        self.min_trust_score = min_trust_score
        self.description_suffix = description_suffix

        # Extract tool metadata
        if isinstance(tool, BaseTool):
            self.name = tool.name
            self.description = tool.description
        elif hasattr(tool, "__name__"):
            self.name = tool.__name__
            self.description = tool.__doc__ or ""
        else:
            self.name = "unknown_tool"
            self.description = ""

        if description_suffix:
            self.description = f"{self.description} {description_suffix}"

    def can_invoke(
        self,
        invoker_card: TrustedAgentCard,
        handshake: TrustHandshake,
    ) -> TrustVerificationResult:
        """Check if an invoker can use this tool.

        Args:
            invoker_card: The invoking agent's card
            handshake: Trust handshake handler

        Returns:
            TrustVerificationResult indicating if invocation is allowed
        """
        return handshake.verify_peer(
            invoker_card,
            required_capabilities=self.required_capabilities,
            min_trust_score=self.min_trust_score,
        )

    def invoke(
        self,
        invoker_card: TrustedAgentCard,
        handshake: TrustHandshake,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Invoke the tool with trust verification.

        Args:
            invoker_card: The invoking agent's card
            handshake: Trust handshake handler
            *args: Arguments to pass to the tool
            **kwargs: Keyword arguments to pass to the tool

        Returns:
            Tool result if verification passes

        Raises:
            PermissionError: If trust verification fails
        """
        verification = self.can_invoke(invoker_card, handshake)

        if not verification.trusted:
            raise PermissionError(
                f"Trust verification failed for tool '{self.name}': {verification.reason}"
            )

        # Execute the underlying tool
        if isinstance(self.tool, BaseTool):
            return self.tool.invoke(*args, **kwargs)
        else:
            return self.tool(*args, **kwargs)


class TrustedToolExecutor:
    """Executor for running tools with trust verification.

    This class manages tool execution with automatic trust verification,
    audit logging, and policy enforcement.
    """

    def __init__(
        self,
        identity: CMVKIdentity,
        policy: Optional[TrustPolicy] = None,
        tools: Optional[List[TrustGatedTool]] = None,
    ):
        """Initialize trusted tool executor.

        Args:
            identity: This executor's identity
            policy: Trust policy to apply
            tools: Initial list of trust-gated tools
        """
        self.identity = identity
        self.policy = policy or TrustPolicy()
        self.handshake = TrustHandshake(identity, policy)
        self._tools: Dict[str, TrustGatedTool] = {}
        self._audit_log: List[ToolInvocationRecord] = []

        if tools:
            for tool in tools:
                self.register_tool(tool)

    def register_tool(self, tool: TrustGatedTool) -> None:
        """Register a trust-gated tool.

        Args:
            tool: The tool to register
        """
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[TrustGatedTool]:
        """Get a registered tool by name.

        Args:
            name: Tool name

        Returns:
            The tool if found, None otherwise
        """
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def invoke(
        self,
        tool: Union[str, TrustGatedTool],
        input_data: Any,
        invoker_card: Optional[TrustedAgentCard] = None,
    ) -> Any:
        """Invoke a tool with trust verification.

        Args:
            tool: Tool name or TrustGatedTool instance
            input_data: Input to pass to the tool
            invoker_card: The invoking agent's card (uses self if not provided)

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool not found
            PermissionError: If trust verification fails
        """
        # Resolve tool
        if isinstance(tool, str):
            gated_tool = self._tools.get(tool)
            if not gated_tool:
                raise ValueError(f"Tool '{tool}' not found")
        else:
            gated_tool = tool

        # Create self-card if no invoker specified
        if invoker_card is None:
            invoker_card = TrustedAgentCard(
                name=self.identity.agent_name,
                description="Self-invocation",
                capabilities=self.identity.capabilities,
                identity=self.identity,
            )
            invoker_card.sign(self.identity)

        # Verify trust
        verification = gated_tool.can_invoke(invoker_card, self.handshake)

        # Create audit record
        record = ToolInvocationRecord(
            tool_name=gated_tool.name,
            invoker_did=invoker_card.identity.did if invoker_card.identity else None,
            timestamp=datetime.now(timezone.utc),
            verified=verification.trusted,
            trust_score=verification.trust_score,
            input_summary=str(input_data)[:200],
            result_summary="",
            warnings=verification.warnings,
        )

        if not verification.trusted:
            record.result_summary = f"BLOCKED: {verification.reason}"
            if self.policy.audit_all_calls:
                self._audit_log.append(record)
            raise PermissionError(
                f"Trust verification failed for tool '{gated_tool.name}': {verification.reason}"
            )

        # Execute tool
        try:
            if isinstance(gated_tool.tool, BaseTool):
                result = gated_tool.tool.invoke(input_data)
            else:
                result = gated_tool.tool(input_data)
            record.result_summary = str(result)[:200]
        except Exception as e:
            record.result_summary = f"ERROR: {str(e)}"
            if self.policy.audit_all_calls:
                self._audit_log.append(record)
            raise

        if self.policy.audit_all_calls:
            self._audit_log.append(record)

        return result

    def get_audit_log(self) -> List[ToolInvocationRecord]:
        """Get the audit log of tool invocations.

        Returns:
            List of invocation records
        """
        return self._audit_log.copy()

    def clear_audit_log(self) -> None:
        """Clear the audit log."""
        self._audit_log.clear()


def create_trust_gated_tool(
    func: Callable,
    required_capabilities: Optional[List[str]] = None,
    min_trust_score: float = 0.7,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> TrustGatedTool:
    """Create a trust-gated tool from a function.

    This is a convenience function for wrapping simple functions
    with trust verification.

    Args:
        func: The function to wrap
        required_capabilities: Capabilities required to use this tool
        min_trust_score: Minimum trust score required
        name: Optional tool name (defaults to function name)
        description: Optional description (defaults to docstring)

    Returns:
        A TrustGatedTool wrapping the function
    """
    tool = StructuredTool.from_function(
        func=func,
        name=name or func.__name__,
        description=description or func.__doc__ or "",
    )

    return TrustGatedTool(
        tool=tool,
        required_capabilities=required_capabilities,
        min_trust_score=min_trust_score,
    )
