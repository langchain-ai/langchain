"""Governance callback handler for tool execution authorization.

Implements structural authority separation (PROPOSE / DECIDE / PROMOTE)
for LangChain tool calls. The handler interposes deterministic policy
evaluation between an agent's tool selection and tool execution.

Example:
    .. code-block:: python

        from langchain_core.callbacks.governance import GovernanceCallbackHandler

        policy = {
            "default": "deny",
            "rules": [
                {
                    "tools": ["search", "wikipedia"],
                    "verdict": "approve",
                },
                {
                    "tools": ["shell", "python_repl"],
                    "verdict": "deny",
                },
            ],
        }
        handler = GovernanceCallbackHandler(policy=policy)
        agent.invoke(inputs, config={"callbacks": [handler]})
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from langchain_core.callbacks import BaseCallbackHandler

if TYPE_CHECKING:
    from uuid import UUID


class ToolExecutionDeniedError(Exception):
    """Raised when a tool call is denied by governance policy."""

    def __init__(self, tool_name: str, reason: str) -> None:
        """Initialize with the denied tool name and reason.

        Args:
            tool_name: Name of the tool that was denied.
            reason: Human-readable denial reason.
        """
        self.tool_name = tool_name
        self.reason = reason
        super().__init__(f"Tool '{tool_name}' denied: {reason}")


# Backwards-compatible alias
ToolExecutionDenied = ToolExecutionDeniedError


class GovernanceCallbackHandler(BaseCallbackHandler):
    """Callback handler that enforces deterministic governance policies on tool calls.

    Implements a three-phase authorization pipeline:

    - **PROPOSE**: Converts each tool call into a structured intent object
      with a SHA-256 content hash.
    - **DECIDE**: Evaluates the intent against user-defined policy rules.
      Pure function — no LLM involvement, no interpretation ambiguity.
    - **PROMOTE**: Allows approved calls, raises ``ToolExecutionDeniedError`` for
      denied calls, and logs every verdict to a hash-chained witness file.

    The handler must be used with ``raise_error=True`` (set by default) so that
    denied tool calls propagate as exceptions and prevent execution.

    Args:
        policy: A dict defining governance rules. Structure::

            {
                "default": "approve" | "deny",
                "rules": [
                    {
                        "tools": ["tool_name", ...],
                        "verdict": "approve" | "deny",
                        "constraints": {  # optional
                            "blocked_patterns": ["rm -rf", ...],
                            "allowed_patterns": ["--dry-run", ...],
                        },
                    },
                    ...
                ],
            }

        witness_path: Path to the witness log file. If ``None``, witness
            logging is disabled. Defaults to ``None``.

    Example:
        .. code-block:: python

            policy = {
                "default": "deny",
                "rules": [
                    {"tools": ["search"], "verdict": "approve"},
                    {"tools": ["shell"], "verdict": "deny"},
                ],
            }

            handler = GovernanceCallbackHandler(
                policy=policy,
                witness_path="./governance_witness.jsonl",
            )

            # Use with any LangChain agent or chain
            agent.invoke(
                {"input": "look up the weather"},
                config={"callbacks": [handler]},
            )
    """

    raise_error: bool = True
    """Must be True for governance to block denied tool calls."""

    def __init__(
        self,
        policy: dict[str, Any],
        witness_path: str | Path | None = None,
    ) -> None:
        """Initialize with governance policy and optional witness log path.

        Args:
            policy: Dict defining governance rules with ``default`` verdict
                and ``rules`` list.
            witness_path: Path to the hash-chained witness log file.
                If ``None``, witness logging is disabled.
        """
        self.policy = policy
        self._prev_hash = "0" * 64

        self._witness_file: Path | None = None
        if witness_path is not None:
            self._witness_file = Path(witness_path)
            self._witness_file.parent.mkdir(parents=True, exist_ok=True)

    # --- PROPOSE phase ---

    @staticmethod
    def _propose(
        serialized: dict[str, Any],
        input_str: str,
        inputs: dict[str, Any] | None,
        tags: list[str] | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Convert a tool call into a structured, hashable intent."""
        intent: dict[str, Any] = {
            "tool": serialized.get("name", "unknown"),
            "input_str": input_str,
        }
        if inputs is not None:
            intent["inputs"] = inputs
        if tags:
            intent["tags"] = tags
        if metadata:
            intent["metadata"] = metadata

        payload = json.dumps(intent, sort_keys=True, default=str).encode()
        intent["content_hash"] = hashlib.sha256(payload).hexdigest()
        return intent

    # --- DECIDE phase ---

    @staticmethod
    def _decide(intent: dict[str, Any], policy: dict[str, Any]) -> str:
        """Pure function: ``(intent, policy) -> 'approve' | 'deny'``.

        No LLM involvement. No interpretation ambiguity.
        """
        tool_name = intent["tool"]
        input_str = intent.get("input_str", "")

        for rule in policy.get("rules", []):
            if tool_name not in rule.get("tools", []):
                continue

            # Check argument constraints before applying verdict
            constraints = rule.get("constraints", {})
            if constraints:
                blocked = constraints.get("blocked_patterns", [])
                for pattern in blocked:
                    if pattern.lower() in input_str.lower():
                        return "deny"

                allowed = constraints.get("allowed_patterns")
                if allowed is not None and not any(
                    p.lower() in input_str.lower() for p in allowed
                ):
                    return "deny"

            return rule.get("verdict", policy.get("default", "deny"))

        return policy.get("default", "deny")

    # --- Witness log ---

    def _record_witness(self, entry: dict[str, Any]) -> None:
        """Append an entry to the hash-chained witness log."""
        if self._witness_file is None:
            return

        entry["prev_hash"] = self._prev_hash
        entry["timestamp"] = time.time()
        payload = json.dumps(entry, sort_keys=True, default=str)
        entry_hash = hashlib.sha256(payload.encode()).hexdigest()
        entry["hash"] = entry_hash

        with self._witness_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        self._prev_hash = entry_hash

    # --- PROMOTE phase (the callback) ---

    @override
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
        """Evaluate tool call against governance policy before execution.

        Args:
            serialized: The serialized tool metadata.
            input_str: String representation of the tool input.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            tags: Tags associated with this run.
            metadata: Metadata associated with this run.
            inputs: Structured tool inputs.
            **kwargs: Additional keyword arguments.

        Raises:
            ToolExecutionDeniedError: If the policy denies this tool call.
        """
        # PROPOSE
        intent = self._propose(serialized, input_str, inputs, tags, metadata)

        # DECIDE
        verdict = self._decide(intent, self.policy)

        # PROMOTE
        self._record_witness({
            "phase": "promote",
            "verdict": verdict,
            "tool": intent["tool"],
            "content_hash": intent["content_hash"],
            "run_id": str(run_id),
        })

        if verdict == "deny":
            raise ToolExecutionDeniedError(
                tool_name=intent["tool"],
                reason="Denied by governance policy",
            )

    @override
    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Log tool completion to witness trail.

        Args:
            output: The tool output.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            **kwargs: Additional keyword arguments.
        """
        output_str = str(output) if output is not None else ""
        self._record_witness({
            "phase": "audit",
            "run_id": str(run_id),
            "result_hash": hashlib.sha256(output_str.encode()).hexdigest(),
        })

    @override
    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Log tool errors to witness trail.

        Args:
            error: The exception that occurred.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            **kwargs: Additional keyword arguments.
        """
        self._record_witness({
            "phase": "error",
            "run_id": str(run_id),
            "error_type": type(error).__name__,
        })


def verify_witness_log(log_path: str | Path) -> bool:
    """Verify the integrity of a governance witness log.

    Each entry in the log contains a ``prev_hash`` linking it to the
    preceding entry and a ``hash`` of its own contents. This function
    walks the chain and checks every link.

    Args:
        log_path: Path to the witness log file.

    Returns:
        ``True`` if the chain is intact, ``False`` if any entry has been
        tampered with or the chain is broken.
    """
    prev_hash = "0" * 64
    path = Path(log_path)

    if not path.exists():
        return True  # Empty log is trivially valid

    with path.open(encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            recorded_hash = entry.pop("hash")

            if entry["prev_hash"] != prev_hash:
                return False

            payload = json.dumps(entry, sort_keys=True, default=str)
            computed = hashlib.sha256(payload.encode()).hexdigest()
            if computed != recorded_hash:
                return False

            prev_hash = recorded_hash

    return True
