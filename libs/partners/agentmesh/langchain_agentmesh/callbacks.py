"""Trust callback handlers for LangChain.

This module provides callback handlers that monitor and enforce
trust policies during chain execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from langchain_agentmesh.identity import CMVKIdentity
from langchain_agentmesh.trust import TrustPolicy, TrustHandshake, TrustedAgentCard


@dataclass
class TrustEvent:
    """A trust-related event during chain execution."""

    event_type: str
    timestamp: datetime
    details: Dict[str, Any]
    trust_score: Optional[float] = None
    verified: bool = True
    warnings: List[str] = field(default_factory=list)


class TrustCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler for trust monitoring and enforcement.

    This handler monitors chain execution and can enforce trust policies,
    log trust-related events, and block untrusted operations.
    """

    def __init__(
        self,
        identity: CMVKIdentity,
        policy: Optional[TrustPolicy] = None,
        peer_cards: Optional[List[TrustedAgentCard]] = None,
    ):
        """Initialize trust callback handler.

        Args:
            identity: This agent's identity
            policy: Trust policy to enforce
            peer_cards: Pre-verified peer agent cards
        """
        self.identity = identity
        self.policy = policy or TrustPolicy()
        self.handshake = TrustHandshake(identity, policy)
        self._events: List[TrustEvent] = []
        self._verified_peers: Dict[str, TrustedAgentCard] = {}

        # Pre-verify peer cards
        if peer_cards:
            for card in peer_cards:
                if card.identity:
                    self._verified_peers[card.identity.did] = card

    def _log_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        trust_score: Optional[float] = None,
        verified: bool = True,
        warnings: Optional[List[str]] = None,
    ) -> TrustEvent:
        """Log a trust event.

        Args:
            event_type: Type of event
            details: Event details
            trust_score: Associated trust score
            verified: Whether the event was verified
            warnings: Any warnings generated

        Returns:
            The logged TrustEvent
        """
        event = TrustEvent(
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            details=details,
            trust_score=trust_score,
            verified=verified,
            warnings=warnings or [],
        )
        self._events.append(event)
        return event

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle LLM start event.

        Args:
            serialized: Serialized LLM configuration
            prompts: List of prompts
            run_id: Unique run identifier
            parent_run_id: Parent run identifier
            tags: Optional tags
            metadata: Optional metadata
            **kwargs: Additional arguments
        """
        if self.policy.audit_all_calls:
            self._log_event(
                "llm_start",
                {
                    "run_id": str(run_id),
                    "model": serialized.get("name", "unknown"),
                    "prompt_count": len(prompts),
                    "has_parent": parent_run_id is not None,
                },
                trust_score=1.0,
                verified=True,
            )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle LLM end event.

        Args:
            response: LLM response
            run_id: Unique run identifier
            parent_run_id: Parent run identifier
            **kwargs: Additional arguments
        """
        if self.policy.audit_all_calls:
            self._log_event(
                "llm_end",
                {
                    "run_id": str(run_id),
                    "generation_count": len(response.generations),
                },
                trust_score=1.0,
                verified=True,
            )

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle tool start event with trust verification.

        Args:
            serialized: Serialized tool configuration
            input_str: Tool input string
            run_id: Unique run identifier
            parent_run_id: Parent run identifier
            tags: Optional tags
            metadata: Optional metadata
            inputs: Tool inputs
            **kwargs: Additional arguments
        """
        tool_name = serialized.get("name", "unknown")
        warnings: List[str] = []

        # Check if tool requires verification
        if self.policy.require_verification:
            # Check if invoker is in verified peers
            invoker_did = metadata.get("invoker_did") if metadata else None

            if invoker_did and invoker_did in self._verified_peers:
                peer_card = self._verified_peers[invoker_did]
                result = self.handshake.verify_peer(peer_card)

                if not result.trusted and self.policy.block_unverified:
                    self._log_event(
                        "tool_blocked",
                        {
                            "tool_name": tool_name,
                            "run_id": str(run_id),
                            "reason": result.reason,
                        },
                        trust_score=result.trust_score,
                        verified=False,
                    )
                    raise PermissionError(
                        f"Tool '{tool_name}' blocked: {result.reason}"
                    )

                warnings = result.warnings
            elif self.policy.block_unverified:
                warnings.append("No verified invoker identity provided")

        self._log_event(
            "tool_start",
            {
                "tool_name": tool_name,
                "run_id": str(run_id),
                "input_length": len(input_str),
            },
            trust_score=1.0 if not warnings else 0.5,
            verified=len(warnings) == 0,
            warnings=warnings,
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle tool end event.

        Args:
            output: Tool output
            run_id: Unique run identifier
            parent_run_id: Parent run identifier
            **kwargs: Additional arguments
        """
        if self.policy.audit_all_calls:
            self._log_event(
                "tool_end",
                {
                    "run_id": str(run_id),
                    "output_type": type(output).__name__,
                },
                trust_score=1.0,
                verified=True,
            )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle tool error event.

        Args:
            error: The error that occurred
            run_id: Unique run identifier
            parent_run_id: Parent run identifier
            **kwargs: Additional arguments
        """
        self._log_event(
            "tool_error",
            {
                "run_id": str(run_id),
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
            trust_score=0.0,
            verified=False,
        )

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle chain start event.

        Args:
            serialized: Serialized chain configuration
            inputs: Chain inputs
            run_id: Unique run identifier
            parent_run_id: Parent run identifier
            tags: Optional tags
            metadata: Optional metadata
            **kwargs: Additional arguments
        """
        if self.policy.audit_all_calls:
            self._log_event(
                "chain_start",
                {
                    "chain_name": serialized.get("name", "unknown"),
                    "run_id": str(run_id),
                    "input_keys": list(inputs.keys()),
                },
                trust_score=1.0,
                verified=True,
            )

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle chain end event.

        Args:
            outputs: Chain outputs
            run_id: Unique run identifier
            parent_run_id: Parent run identifier
            **kwargs: Additional arguments
        """
        if self.policy.audit_all_calls:
            self._log_event(
                "chain_end",
                {
                    "run_id": str(run_id),
                    "output_keys": list(outputs.keys()),
                },
                trust_score=1.0,
                verified=True,
            )

    def on_agent_action(
        self,
        action: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle agent action event.

        Args:
            action: The agent action
            run_id: Unique run identifier
            parent_run_id: Parent run identifier
            **kwargs: Additional arguments
        """
        if self.policy.audit_all_calls:
            self._log_event(
                "agent_action",
                {
                    "run_id": str(run_id),
                    "action_type": type(action).__name__,
                    "tool": getattr(action, "tool", "unknown"),
                },
                trust_score=1.0,
                verified=True,
            )

    def on_agent_finish(
        self,
        finish: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle agent finish event.

        Args:
            finish: The agent finish result
            run_id: Unique run identifier
            parent_run_id: Parent run identifier
            **kwargs: Additional arguments
        """
        if self.policy.audit_all_calls:
            self._log_event(
                "agent_finish",
                {
                    "run_id": str(run_id),
                },
                trust_score=1.0,
                verified=True,
            )

    def add_verified_peer(self, card: TrustedAgentCard) -> bool:
        """Add a verified peer agent.

        Args:
            card: The peer's agent card

        Returns:
            True if peer was verified and added, False otherwise
        """
        result = self.handshake.verify_peer(card)
        if result.trusted and card.identity:
            self._verified_peers[card.identity.did] = card
            return True
        return False

    def remove_peer(self, did: str) -> bool:
        """Remove a peer from verified list.

        Args:
            did: The peer's DID

        Returns:
            True if peer was removed, False if not found
        """
        if did in self._verified_peers:
            del self._verified_peers[did]
            return True
        return False

    def get_events(self) -> List[TrustEvent]:
        """Get all logged trust events.

        Returns:
            List of trust events
        """
        return self._events.copy()

    def get_events_by_type(self, event_type: str) -> List[TrustEvent]:
        """Get events filtered by type.

        Args:
            event_type: Type of events to retrieve

        Returns:
            List of matching events
        """
        return [e for e in self._events if e.event_type == event_type]

    def get_unverified_events(self) -> List[TrustEvent]:
        """Get all unverified events.

        Returns:
            List of unverified events
        """
        return [e for e in self._events if not e.verified]

    def clear_events(self) -> None:
        """Clear all logged events."""
        self._events.clear()

    def get_trust_summary(self) -> Dict[str, Any]:
        """Get a summary of trust-related activity.

        Returns:
            Dictionary with trust metrics
        """
        total = len(self._events)
        verified = sum(1 for e in self._events if e.verified)
        unverified = total - verified

        event_types: Dict[str, int] = {}
        for event in self._events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1

        return {
            "total_events": total,
            "verified_events": verified,
            "unverified_events": unverified,
            "verification_rate": verified / total if total > 0 else 1.0,
            "event_types": event_types,
            "verified_peers": len(self._verified_peers),
            "warnings_count": sum(len(e.warnings) for e in self._events),
        }
