"""Tests for GovernanceCallbackHandler."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from pathlib import Path
    from uuid import UUID

import pytest

from langchain_core.callbacks.governance import (
    GovernanceCallbackHandler,
    ToolExecutionDeniedError,
    verify_witness_log,
)


@pytest.fixture
def run_id() -> UUID:
    return uuid4()


@pytest.fixture
def witness_path(tmp_path: Path) -> Path:
    return tmp_path / "witness.jsonl"


@pytest.fixture
def deny_by_default_policy() -> dict:
    return {
        "default": "deny",
        "rules": [
            {"tools": ["search", "wikipedia"], "verdict": "approve"},
            {"tools": ["shell", "python_repl"], "verdict": "deny"},
        ],
    }


@pytest.fixture
def approve_by_default_policy() -> dict:
    return {
        "default": "approve",
        "rules": [
            {"tools": ["shell"], "verdict": "deny"},
        ],
    }


class TestPropose:
    def test_creates_intent_with_hash(self) -> None:
        intent = GovernanceCallbackHandler._propose(
            serialized={"name": "search"},
            input_str="weather in London",
            inputs={"query": "weather in London"},
            tags=None,
            metadata=None,
        )
        assert intent["tool"] == "search"
        assert intent["input_str"] == "weather in London"
        assert "content_hash" in intent
        assert len(intent["content_hash"]) == 64  # SHA-256 hex

    def test_deterministic_hash(self) -> None:
        h1 = GovernanceCallbackHandler._propose(
            serialized={"name": "search"},
            input_str="test",
            inputs=None,
            tags=None,
            metadata=None,
        )["content_hash"]
        h2 = GovernanceCallbackHandler._propose(
            serialized={"name": "search"},
            input_str="test",
            inputs=None,
            tags=None,
            metadata=None,
        )["content_hash"]
        assert h1 == h2

    def test_different_inputs_different_hash(self) -> None:
        h1 = GovernanceCallbackHandler._propose(
            {"name": "search"}, "a", None, None, None
        )["content_hash"]
        h2 = GovernanceCallbackHandler._propose(
            {"name": "search"}, "b", None, None, None
        )["content_hash"]
        assert h1 != h2

    def test_unknown_tool_name(self) -> None:
        intent = GovernanceCallbackHandler._propose(
            serialized={}, input_str="test", inputs=None, tags=None, metadata=None
        )
        assert intent["tool"] == "unknown"


class TestDecide:
    def test_approve_allowed_tool(self, deny_by_default_policy: dict) -> None:
        intent = {"tool": "search", "input_str": "test"}
        assert (
            GovernanceCallbackHandler._decide(intent, deny_by_default_policy)
            == "approve"
        )

    def test_deny_blocked_tool(self, deny_by_default_policy: dict) -> None:
        intent = {"tool": "shell", "input_str": "ls"}
        assert (
            GovernanceCallbackHandler._decide(intent, deny_by_default_policy) == "deny"
        )

    def test_default_deny_unknown_tool(self, deny_by_default_policy: dict) -> None:
        intent = {"tool": "unknown_tool", "input_str": "test"}
        assert (
            GovernanceCallbackHandler._decide(intent, deny_by_default_policy) == "deny"
        )

    def test_default_approve_unknown_tool(
        self, approve_by_default_policy: dict
    ) -> None:
        intent = {"tool": "unknown_tool", "input_str": "test"}
        assert (
            GovernanceCallbackHandler._decide(intent, approve_by_default_policy)
            == "approve"
        )

    def test_blocked_pattern_constraint(self) -> None:
        policy = {
            "default": "deny",
            "rules": [
                {
                    "tools": ["shell"],
                    "verdict": "approve",
                    "constraints": {"blocked_patterns": ["rm -rf", "sudo"]},
                },
            ],
        }
        assert (
            GovernanceCallbackHandler._decide(
                {"tool": "shell", "input_str": "ls -la"}, policy
            )
            == "approve"
        )
        assert (
            GovernanceCallbackHandler._decide(
                {"tool": "shell", "input_str": "sudo rm -rf /"}, policy
            )
            == "deny"
        )

    def test_allowed_pattern_constraint(self) -> None:
        policy = {
            "default": "deny",
            "rules": [
                {
                    "tools": ["shell"],
                    "verdict": "approve",
                    "constraints": {"allowed_patterns": ["--dry-run"]},
                },
            ],
        }
        assert (
            GovernanceCallbackHandler._decide(
                {"tool": "shell", "input_str": "deploy --dry-run"}, policy
            )
            == "approve"
        )
        assert (
            GovernanceCallbackHandler._decide(
                {"tool": "shell", "input_str": "deploy --force"}, policy
            )
            == "deny"
        )

    def test_empty_policy(self) -> None:
        assert (
            GovernanceCallbackHandler._decide(
                {"tool": "anything", "input_str": ""}, {"default": "deny"}
            )
            == "deny"
        )

    def test_case_insensitive_pattern_matching(self) -> None:
        policy = {
            "default": "deny",
            "rules": [
                {
                    "tools": ["shell"],
                    "verdict": "approve",
                    "constraints": {"blocked_patterns": ["sudo"]},
                },
            ],
        }
        assert (
            GovernanceCallbackHandler._decide(
                {"tool": "shell", "input_str": "SUDO reboot"}, policy
            )
            == "deny"
        )


class TestPromote:
    def test_approved_tool_does_not_raise(
        self, deny_by_default_policy: dict, run_id: Any
    ) -> None:
        handler = GovernanceCallbackHandler(policy=deny_by_default_policy)
        # Should not raise
        handler.on_tool_start(
            serialized={"name": "search"},
            input_str="weather",
            run_id=run_id,
        )

    def test_denied_tool_raises(
        self, deny_by_default_policy: dict, run_id: Any
    ) -> None:
        handler = GovernanceCallbackHandler(policy=deny_by_default_policy)
        with pytest.raises(ToolExecutionDeniedError, match="shell"):
            handler.on_tool_start(
                serialized={"name": "shell"},
                input_str="ls",
                run_id=run_id,
            )

    def test_unknown_tool_denied_by_default(
        self, deny_by_default_policy: dict, run_id: Any
    ) -> None:
        handler = GovernanceCallbackHandler(policy=deny_by_default_policy)
        with pytest.raises(ToolExecutionDeniedError):
            handler.on_tool_start(
                serialized={"name": "unknown"},
                input_str="test",
                run_id=run_id,
            )

    def test_raise_error_is_true_by_default(self) -> None:
        handler = GovernanceCallbackHandler(policy={"default": "approve"})
        assert handler.raise_error is True


class TestWitnessLog:
    def test_witness_log_created(
        self, deny_by_default_policy: dict, witness_path: Path, run_id: Any
    ) -> None:
        handler = GovernanceCallbackHandler(
            policy=deny_by_default_policy, witness_path=witness_path
        )
        handler.on_tool_start(
            serialized={"name": "search"}, input_str="test", run_id=run_id
        )
        assert witness_path.exists()
        entries = [json.loads(line) for line in witness_path.read_text().splitlines()]
        assert len(entries) == 1
        assert entries[0]["verdict"] == "approve"
        assert entries[0]["phase"] == "promote"

    def test_denied_tool_logged_before_raise(
        self, deny_by_default_policy: dict, witness_path: Path, run_id: Any
    ) -> None:
        handler = GovernanceCallbackHandler(
            policy=deny_by_default_policy, witness_path=witness_path
        )
        with pytest.raises(ToolExecutionDeniedError):
            handler.on_tool_start(
                serialized={"name": "shell"}, input_str="ls", run_id=run_id
            )
        entries = [json.loads(line) for line in witness_path.read_text().splitlines()]
        assert len(entries) == 1
        assert entries[0]["verdict"] == "deny"

    def test_hash_chain_integrity(
        self, deny_by_default_policy: dict, witness_path: Path, run_id: Any
    ) -> None:
        handler = GovernanceCallbackHandler(
            policy=deny_by_default_policy, witness_path=witness_path
        )
        # Generate multiple entries
        handler.on_tool_start(
            serialized={"name": "search"}, input_str="q1", run_id=run_id
        )
        handler.on_tool_end(output="result1", run_id=run_id)
        handler.on_tool_start(
            serialized={"name": "wikipedia"}, input_str="q2", run_id=run_id
        )
        assert verify_witness_log(witness_path) is True

    def test_tampered_log_detected(
        self, deny_by_default_policy: dict, witness_path: Path, run_id: Any
    ) -> None:
        handler = GovernanceCallbackHandler(
            policy=deny_by_default_policy, witness_path=witness_path
        )
        handler.on_tool_start(
            serialized={"name": "search"}, input_str="q1", run_id=run_id
        )
        handler.on_tool_start(
            serialized={"name": "search"}, input_str="q2", run_id=run_id
        )

        # Tamper with the log
        lines = witness_path.read_text().splitlines()
        entry = json.loads(lines[0])
        entry["verdict"] = "tampered"
        lines[0] = json.dumps(entry)
        witness_path.write_text("\n".join(lines) + "\n")

        assert verify_witness_log(witness_path) is False

    def test_on_tool_end_logs_result_hash(
        self, deny_by_default_policy: dict, witness_path: Path, run_id: Any
    ) -> None:
        handler = GovernanceCallbackHandler(
            policy=deny_by_default_policy, witness_path=witness_path
        )
        handler.on_tool_start(
            serialized={"name": "search"}, input_str="test", run_id=run_id
        )
        handler.on_tool_end(output="some result", run_id=run_id)
        entries = [json.loads(line) for line in witness_path.read_text().splitlines()]
        assert entries[1]["phase"] == "audit"
        assert "result_hash" in entries[1]

    def test_on_tool_error_logs(
        self, deny_by_default_policy: dict, witness_path: Path, run_id: Any
    ) -> None:
        handler = GovernanceCallbackHandler(
            policy=deny_by_default_policy, witness_path=witness_path
        )
        handler.on_tool_start(
            serialized={"name": "search"}, input_str="test", run_id=run_id
        )
        handler.on_tool_error(error=ValueError("test"), run_id=run_id)
        entries = [json.loads(line) for line in witness_path.read_text().splitlines()]
        assert entries[1]["phase"] == "error"
        assert entries[1]["error_type"] == "ValueError"

    def test_no_witness_path_does_not_error(
        self, deny_by_default_policy: dict, run_id: Any
    ) -> None:
        handler = GovernanceCallbackHandler(policy=deny_by_default_policy)
        handler.on_tool_start(
            serialized={"name": "search"}, input_str="test", run_id=run_id
        )
        # No error, just no log file

    def test_verify_empty_log(self, tmp_path: Path) -> None:
        assert verify_witness_log(tmp_path / "nonexistent.jsonl") is True

    def test_genesis_hash(
        self, deny_by_default_policy: dict, witness_path: Path, run_id: Any
    ) -> None:
        handler = GovernanceCallbackHandler(
            policy=deny_by_default_policy, witness_path=witness_path
        )
        handler.on_tool_start(
            serialized={"name": "search"}, input_str="test", run_id=run_id
        )
        entry = json.loads(witness_path.read_text().splitlines()[0])
        assert entry["prev_hash"] == "0" * 64


class TestToolExecutionDeniedError:
    def test_exception_message(self) -> None:
        exc = ToolExecutionDeniedError("shell", "blocked by policy")
        assert "shell" in str(exc)
        assert "blocked by policy" in str(exc)

    def test_exception_attributes(self) -> None:
        exc = ToolExecutionDeniedError("shell", "test reason")
        assert exc.tool_name == "shell"
        assert exc.reason == "test reason"
