"""Joy Trust callback handler for LangChain."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from langchain_joy.client import JoyTrustClient, JoyTrustError

if TYPE_CHECKING:
    from langchain_core.agents import AgentAction, AgentFinish


logger = logging.getLogger(__name__)


class JoyTrustVerificationError(Exception):
    """Raised when trust verification fails."""

    def __init__(
        self,
        message: str,
        *,
        agent_id: str | None = None,
        trust_score: float | None = None,
        threshold: float | None = None,
    ) -> None:
        """Initialize verification error.

        Args:
            message: Error message.
            agent_id: The agent that failed verification.
            trust_score: The agent's trust score.
            threshold: The required threshold.
        """
        super().__init__(message)
        self.agent_id = agent_id
        self.trust_score = trust_score
        self.threshold = threshold


class JoyTrustCallbackHandler(BaseCallbackHandler):
    """Callback handler that verifies agent trust before tool execution.

    This handler intercepts on_tool_start events and verifies the tool/agent
    against Joy's trust network before allowing execution.

    Example:
        >>> handler = JoyTrustCallbackHandler(min_trust_score=1.5)
        >>> agent = initialize_agent(tools, llm, callbacks=[handler])
        >>> agent.run("Use the calculator")  # Verifies calculator agent trust
    """

    def __init__(
        self,
        *,
        min_trust_score: float = 1.5,
        fail_open: bool = False,
        api_key: str | None = None,
        cache_ttl: int = 300,
    ) -> None:
        """Initialize Joy Trust callback handler.

        Args:
            min_trust_score: Minimum trust score required (0-5 scale).
            fail_open: If True, allow on errors. If False, block on errors.
            api_key: Optional API key for higher rate limits.
            cache_ttl: Cache TTL in seconds.
        """
        self.min_trust_score = min_trust_score
        self.fail_open = fail_open
        self.client = JoyTrustClient(api_key=api_key, cache_ttl=cache_ttl)
        self._verification_log: list[dict[str, Any]] = []

    def _extract_agent_id(
        self,
        tool_name: str,
        tool_input: dict[str, Any] | str,
    ) -> str | None:
        """Extract agent ID from tool metadata.

        Looks for agent_id in:
        1. tool_input dict if it has agent_id key
        2. tool_name if it looks like an agent ID

        Args:
            tool_name: Name of the tool being called.
            tool_input: Input to the tool.

        Returns:
            Agent ID if found, None otherwise.
        """
        # Check tool_input for agent_id
        if isinstance(tool_input, dict):
            for key in ["agent_id", "agentId", "target_agent", "delegate_to"]:
                if key in tool_input:
                    return str(tool_input[key])

        # Check if tool_name is an agent ID
        if tool_name.startswith("ag_"):
            return tool_name

        return None

    def _log_verification(
        self,
        *,
        tool_name: str,
        agent_id: str | None,
        trust_score: float | None,
        allowed: bool,
        reason: str,
    ) -> None:
        """Log verification attempt for audit."""
        import time

        entry = {
            "timestamp": time.time(),
            "tool_name": tool_name,
            "agent_id": agent_id,
            "trust_score": trust_score,
            "threshold": self.min_trust_score,
            "allowed": allowed,
            "reason": reason,
        }
        self._verification_log.append(entry)
        logger.info(
            "Joy trust verification: tool=%s agent=%s score=%s allowed=%s reason=%s",
            tool_name,
            agent_id,
            trust_score,
            allowed,
            reason,
        )

    def get_verification_log(self) -> list[dict[str, Any]]:
        """Get the verification audit log.

        Returns:
            List of verification entries with timestamps.
        """
        return self._verification_log.copy()

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Verify trust before tool execution.

        Args:
            serialized: Serialized tool information.
            input_str: String input to the tool.
            run_id: Unique run identifier.
            parent_run_id: Parent run identifier.
            tags: Optional tags.
            metadata: Optional metadata.
            inputs: Optional structured inputs.
            **kwargs: Additional arguments.

        Raises:
            JoyTrustVerificationError: If trust verification fails.
        """
        tool_name = serialized.get("name", "unknown")

        # Try to extract agent ID from inputs or metadata
        agent_id = None
        if inputs:
            agent_id = self._extract_agent_id(tool_name, inputs)
        if not agent_id and metadata:
            agent_id = metadata.get("agent_id") or metadata.get("target_agent")

        # If no agent ID found, skip verification
        if not agent_id:
            self._log_verification(
                tool_name=tool_name,
                agent_id=None,
                trust_score=None,
                allowed=True,
                reason="no_agent_id",
            )
            return

        # Verify trust
        try:
            result = self.client.verify_trust(agent_id, min_trust=self.min_trust_score)
            trust_score = result["trust_score"]
            meets_threshold = result["meets_threshold"]

            if meets_threshold:
                self._log_verification(
                    tool_name=tool_name,
                    agent_id=agent_id,
                    trust_score=trust_score,
                    allowed=True,
                    reason="trust_verified",
                )
            else:
                self._log_verification(
                    tool_name=tool_name,
                    agent_id=agent_id,
                    trust_score=trust_score,
                    allowed=False,
                    reason="below_threshold",
                )
                raise JoyTrustVerificationError(
                    f"Agent {agent_id} trust score {trust_score} "
                    f"below threshold {self.min_trust_score}",
                    agent_id=agent_id,
                    trust_score=trust_score,
                    threshold=self.min_trust_score,
                )

        except JoyTrustError as e:
            if self.fail_open:
                self._log_verification(
                    tool_name=tool_name,
                    agent_id=agent_id,
                    trust_score=None,
                    allowed=True,
                    reason=f"error_fail_open: {e}",
                )
            else:
                self._log_verification(
                    tool_name=tool_name,
                    agent_id=agent_id,
                    trust_score=None,
                    allowed=False,
                    reason=f"error_fail_closed: {e}",
                )
                raise JoyTrustVerificationError(
                    f"Trust verification failed for {agent_id}: {e}",
                    agent_id=agent_id,
                ) from e

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Log agent actions for audit trail.

        Args:
            action: The agent action being taken.
            run_id: Unique run identifier.
            parent_run_id: Parent run identifier.
            tags: Optional tags.
            **kwargs: Additional arguments.
        """
        logger.debug(
            "Agent action: tool=%s input=%s",
            action.tool,
            action.tool_input,
        )

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Log agent completion.

        Args:
            finish: The agent finish event.
            run_id: Unique run identifier.
            parent_run_id: Parent run identifier.
            tags: Optional tags.
            **kwargs: Additional arguments.
        """
        logger.debug("Agent finished: %s", finish.return_values)
