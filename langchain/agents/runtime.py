"""Agent runtime management: track metadata, state, and lifecycle for agents.

Similar to ToolRuntime, it provides static configuration (metadata) and dynamic
runtime state management for agents, supporting integration with existing Agent
classes for easy monitoring, debugging, and extension.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, Optional, TypeVar

# Type variable for compatibility with different Agent types
T = TypeVar("T")


class AgentStatus(str, Enum):
    """Agent runtime status enumeration (for monitoring).

    Includes five states: initialized, running, succeeded, failed, and idle.
    """

    INITIALIZED = "initialized"  # Initialized
    RUNNING = "running"  # In operation
    SUCCEEDED = "succeeded"  # Task succeeded
    FAILED = "failed"  # Task failed
    IDLE = "idle"  # Idle state


@dataclass(frozen=True)
class AgentMetadata:
    """Static metadata for agents (immutable configuration).

    Supports serialization, including name, description, type, and other
    information.
    """

    name: str
    """Agent name (unique identifier, e.g., "search_agent")"""
    description: str
    """Agent function description (used for selection during
    automatic planning)"""
    agent_type: str
    """Agent type (e.g., "structured_chat", "react")"""
    model_name: str
    """Associated LLM model name (e.g., "gpt-4")"""
    created_at: datetime = datetime.now()
    """Creation time (default: when initialized)"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (supports JSON serialization)"""
        data = asdict(self)
        # Convert datetime to ISO string
        data["created_at"] = self.created_at.isoformat()
        return data


class AgentState(Generic[T]):
    """Dynamic runtime state of agents (mutable, updated during execution)"""

    def __init__(self):
        self.status: AgentStatus = AgentStatus.INITIALIZED
        self.call_count: int = 0  # Total number of calls
        self.last_call_time: Optional[datetime] = None  # Time of last call
        self.error: Optional[str] = None  # Latest error message
        # Context data (generic, supports any type)
        self.context: Optional[T] = None

    def update_on_call(self) -> None:
        """Update state on agent call (increment count, update time,
        reset error)"""
        self.call_count += 1
        self.last_call_time = datetime.now()
        self.status = AgentStatus.RUNNING
        self.error = None  # Reset error on new call

    def update_on_complete(self, success: bool, error: Optional[str] = None) -> None:
        """Update state on call completion (set success/failure)"""
        if success:
            self.status = AgentStatus.SUCCEEDED
        else:
            self.status = AgentStatus.FAILED
            self.error = error or "Unknown error"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for monitoring output)"""
        last_call_iso = self.last_call_time.isoformat() if self.last_call_time else None
        context_type = type(self.context).__name__ if self.context else None
        return {
            "status": self.status.value,
            "call_count": self.call_count,
            "last_call_time": last_call_iso,
            "error": self.error,
            "context_type": context_type,
        }


class AgentRuntime:
    """Agent runtime manager (integrates metadata and state).

    Provides lifecycle interfaces.
    """

    def __init__(self, metadata: AgentMetadata):
        """Initialize AgentRuntime.

        Associate metadata, and initialize state (default: INITIALIZED).
        """
        self.metadata = metadata
        self.state = AgentState()

    def on_call(self) -> None:
        """Triggered when the agent is called (updates state)"""
        self.state.update_on_call()

    def on_complete(self, success: bool, error: Optional[str] = None) -> None:
        """Triggered when the agent call completes (updates state)"""
        self.state.update_on_complete(success, error)

    def get_summary(self) -> Dict[str, Any]:
        """Get runtime summary information (metadata + state)"""
        return {
            "metadata": self.metadata.to_dict(),
            "state": self.state.to_dict(),
        }

    def __repr__(self) -> str:
        return (
            f"AgentRuntime(agent_name={self.metadata.name!r}, "
            f"status={self.state.status!r})"
        )
