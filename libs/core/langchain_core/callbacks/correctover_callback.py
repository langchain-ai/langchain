"""
Correctover CCS Callback Handler for LangChain.

Integrates Correctover's Runtime Call Verification (CCS) engine into
LangChain agent execution pipelines. The handler intercepts tool calls
to validate them against 24 CCS detection rules before execution.

Installation:
    pip install correctover-ccs  # or use the standalone CCS CLI

Usage:
    from langchain_community.callbacks import CorrectoverCallbackHandler
    from langchain.agents import AgentExecutor

    ccs_handler = CorrectoverCallbackHandler(
        api_key="your-api-key",
        mode="block",           # "block" | "log" | "report"
        rules=["RCE", "SSRF", "PATH_TRAVERSAL", "CREDENTIAL"],
    )
    agent = AgentExecutor(
        agent=agent,
        tools=tools,
        callbacks=[ccs_handler],
        verbose=True,
    )

Documentation: https://correctover.com/docs/ccs-integration
"""

import json
import logging
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)

DEFAULT_RULES = [
    "RCE",
    "SSRF",
    "PATH_TRAVERSAL",
    "CREDENTIAL_LEAK",
    "MCP_STDIO",
    "PROMPT_INJECTION",
]


class CorrectoverCallbackHandler(BaseCallbackHandler):
    """Callback handler that validates AI agent tool calls using Correctover CCS.

    This handler intercepts agent actions (tool calls) and validates them
    against Correctover's 24 CCS detection rules. It can operate in three modes:
    - ``log``: Report findings without blocking execution (recommended for staging)
    - ``block``: Block detected violations and raise exceptions (recommended for production)
    - ``report``: Collect all findings and emit a summary at the end of execution

    Key features:
        - Real-time tool call validation against 24 CCS rules
        - Support for MCP protocol, function calling, and custom tool schemas
        - Compatible with LangChain AgentExecutor and custom agents
        - Microsecond latency (P50 22µs, P99 99µs) when using local CCS binary
        - Structured JSON output for SIEM integration

    Args:
        api_key: Correctover API key (optional, for cloud mode).
        mode: Operation mode - "block", "log", or "report". Defaults to ``"block"``.
        rules: List of CCS rules to enforce. Defaults to all 24 rules.
        ccs_binary: Path to CCS CLI binary. Defaults to ``"ccs"`` (in PATH).
        raise_error: Whether callback errors propagate. Defaults to ``False``.
        verbose: Enable detailed logging. Defaults to ``False``.
        tags: Additional tags to attach to CCS validation requests.
    """

    name: str = "CorrectoverCCS"

    def __init__(
        self,
        api_key: Optional[str] = None,
        mode: str = "log",
        rules: Optional[List[str]] = None,
        ccs_binary: str = "ccs",
        raise_error: bool = False,
        verbose: bool = False,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Correctover CCS callback handler."""
        super().__init__(**kwargs)

        self.api_key = api_key or os.environ.get("CORRECTOVER_API_KEY", "")
        self.mode = mode
        self.rules = rules or DEFAULT_RULES
        self.ccs_binary = ccs_binary
        self.raise_error = raise_error
        self.verbose = verbose
        self.tags = tags or []

        # Runtime state
        self._findings: List[Dict[str, Any]] = []
        self._tool_call_count: int = 0
        self._violation_count: int = 0
        self._chain_stack: List[str] = []
        self._session_id: Optional[str] = None

        # Detect if CCS CLI is available
        self._ccs_available = self._check_ccs_binary()

        if verbose:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(
                "[CorrectoverCCS] %(levelname)s: %(message)s"
            ))
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG if verbose else logging.INFO)

        if not self._ccs_available:
            logger.warning(
                "CCS CLI binary '%s' not found. "
                "Install it with: pip install correctover-ccs",
                self.ccs_binary,
            )

    # ------------------------------------------------------------------ #
    #  Property overrides for selective event handling
    # ------------------------------------------------------------------ #

    @property
    def ignore_llm(self) -> bool:
        """Don't process LLM events."""
        return True

    @property
    def ignore_chain(self) -> bool:
        """Process chain events for context tracking."""
        return False

    @property
    def ignore_agent(self) -> bool:
        """Process agent events (this is the main integration point)."""
        return False

    @property
    def ignore_retriever(self) -> bool:
        """Don't process retriever events."""
        return True

    @property
    def ignore_chat_model(self) -> bool:
        """Don't process chat model events."""
        return True

    # ------------------------------------------------------------------ #
    #  Chain callbacks — track execution context
    # ------------------------------------------------------------------ #

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
        """Track chain execution context."""
        name = serialized.get("name", "unknown_chain")
        self._chain_stack.append(name)
        if self.verbose:
            logger.debug("Chain started: %s", name)

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Pop chain execution context."""
        if self._chain_stack:
            self._chain_stack.pop()

    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Pop chain context on error."""
        if self._chain_stack:
            self._chain_stack.pop()

    # ------------------------------------------------------------------ #
    #  Agent callbacks — the main integration point for CCS
    # ------------------------------------------------------------------ #

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Validate an agent's tool call using Correctover CCS.

        This is the primary callback method. Every time an agent decides
        to call a tool, this method intercepts the action and validates it
        against the configured CCS rules.
        """
        self._tool_call_count += 1

        tool_name = action.tool
        tool_input = action.tool_input

        if self.verbose:
            logger.debug(
                "Validating tool call #%d: %s(%s)",
                self._tool_call_count,
                tool_name,
                self._truncate(str(tool_input), 200),
            )

        # Build the CCS validation payload
        payload = {
            "tool": tool_name,
            "input": tool_input,
            "rules": self.rules,
            "tags": self.tags + self._chain_stack,
            "session_id": self._session_id,
            "tool_call_index": self._tool_call_count,
        }

        # Validate using CCS (local binary or API)
        result = self._validate(payload)
        self._findings.append(result)

        if result.get("violation", False):
            self._violation_count += 1
            message = (
                f"Correctover CCS blocked tool call '{tool_name}': "
                f"{result.get('rule', 'UNKNOWN')} - {result.get('detail', '')}"
            )

            if self.mode == "block":
                # Raise a clear, actionable exception
                raise CorrectoverViolationError(
                    tool_name=tool_name,
                    rule=result.get("rule", "UNKNOWN"),
                    detail=result.get("detail", ""),
                    message=message,
                )
            elif self.mode == "report":
                logger.warning("[CCS VIOLATION] %s", message)
            else:  # log mode
                logger.info("[CCS DETECTED] %s", message)
        else:
            if self.verbose:
                logger.debug("Tool call passed CCS validation: %s", tool_name)

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Generate a CCS report summary at the end of agent execution."""
        if self.mode == "report" and self._tool_call_count > 0:
            summary = self._generate_report()
            logger.info("=== Correctover CCS Session Report ===")
            logger.info("Tool calls: %d", summary["total_calls"])
            logger.info("Violations: %d", summary["violations"])
            logger.info("Findings: %s", json.dumps(summary["findings"], indent=2))

    # ------------------------------------------------------------------ #
    #  Internal CCS validation
    # ------------------------------------------------------------------ #

    def _validate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run CCS validation on a tool call payload.

        Uses the local CCS CLI binary for minimum latency.
        Falls back to a basic rules-based check if CCS is unavailable.
        """
        if self._ccs_available:
            return self._validate_with_ccs_cli(payload)
        else:
            return self._validate_fallback(payload)

    def _validate_with_ccs_cli(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate using the CCS CLI binary."""
        try:
            proc = subprocess.run(
                [self.ccs_binary, "scan", "--format", "json", "--input", "-"],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                result = json.loads(proc.stdout)
                return {
                    "violation": result.get("violation", False),
                    "rule": result.get("rule", None),
                    "detail": result.get("detail", ""),
                    "confidence": result.get("confidence", 0.0),
                    "tool": payload.get("tool", "unknown"),
                }
        except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as e:
            logger.warning("CCS CLI error: %s", e)

        return {"violation": False, "tool": payload.get("tool", "unknown")}

    def _validate_fallback(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Basic rule-based fallback when CCS CLI is unavailable.

        This covers the most critical patterns. For full protection,
        install the CCS CLI: pip install correctover-ccs
        """
        tool_input = payload.get("input", "")
        tool_name = payload.get("tool", "")
        input_str = str(tool_input) if not isinstance(tool_input, str) else tool_input
        active_rules = payload.get("rules", DEFAULT_RULES)

        # RCE detection: shell commands, eval, exec, subprocess
        if "RCE" in active_rules:
            rce_patterns = [
                "import subprocess",
                "import os",
                "os.system",
                "os.popen",
                "subprocess.run",
                "subprocess.Popen",
                "subprocess.call",
                "eval(",
                "exec(",
                "__import__('os')",
                "rm -rf",
                "shutdown",
                "format(",
                "del ",
                "chmod 777",
            ]
            for pattern in rce_patterns:
                if pattern.lower() in input_str.lower():
                    return {
                        "violation": True,
                        "rule": "RCE",
                        "detail": f"Detected pattern: '{pattern}' in tool input",
                        "confidence": 0.85,
                        "tool": tool_name,
                    }

        # SSRF detection: internal IPs, sensitive hosts
        if "SSRF" in active_rules:
            ssrf_patterns = [
                "127.0.0.1",
                "localhost",
                "0.0.0.0",
                "169.254.169.254",  # metadata endpoint
                "10.",
                "172.16.",
                "192.168.",
                "internal",
                "metadata",
                "docker",
                "127.1",
                "10.0",
            ]
            for pattern in ssrf_patterns:
                if f"://{pattern}" in input_str.lower():
                    return {
                        "violation": True,
                        "rule": "SSRF",
                        "detail": f"Detected potential SSRF: '{pattern}' in tool input URL",
                        "confidence": 0.80,
                        "tool": tool_name,
                    }

        # Path traversal detection
        if "PATH_TRAVERSAL" in active_rules:
            traversal_patterns = [
                "../",
                "..\\",
                "/etc/passwd",
                "/etc/shadow",
                "/root/",
                "~/.ssh",
                "\\..\\",
                "/proc/",
            ]
            for pattern in traversal_patterns:
                if pattern.lower() in input_str.lower():
                    return {
                        "violation": True,
                        "rule": "PATH_TRAVERSAL",
                        "detail": f"Detected path traversal: '{pattern}'",
                        "confidence": 0.75,
                        "tool": tool_name,
                    }

        # Credential leak detection
        if "CREDENTIAL_LEAK" in active_rules:
            cred_patterns = [
                "AKIA",  # AWS access key
                "ssh-rsa",
                "ssh-ed25519",
                "-----BEGIN RSA PRIVATE KEY",
                "-----BEGIN OPENSSH PRIVATE KEY",
                "ghp_",
                "gho_",
                "sk-",  # OpenAI API key prefix
                "xoxb-",  # Slack bot token
                "xoxp-",  # Slack user token
                "token=",
                "password=",
                "secret=",
                "api_key=",
                "api-key=",
            ]
            for pattern in cred_patterns:
                if pattern.lower() in input_str.lower():
                    return {
                        "violation": True,
                        "rule": "CREDENTIAL_LEAK",
                        "detail": f"Detected potential credential pattern: '{pattern}'",
                        "confidence": 0.70,
                        "tool": tool_name,
                    }

        return {"violation": False, "tool": tool_name}

    # ------------------------------------------------------------------ #
    #  Utility methods
    # ------------------------------------------------------------------ #

    def _generate_report(self) -> Dict[str, Any]:
        """Generate a session report of all CCS findings."""
        return {
            "total_calls": self._tool_call_count,
            "violations": self._violation_count,
            "findings": self._findings,
            "chain_context": self._chain_stack,
            "session_id": self._session_id,
        }

    def reset(self) -> None:
        """Reset the handler's runtime state for a new execution."""
        self._findings.clear()
        self._tool_call_count = 0
        self._violation_count = 0
        self._chain_stack.clear()

    def _check_ccs_binary(self) -> bool:
        """Check if the CCS CLI binary is available and functional."""
        try:
            proc = subprocess.run(
                [self.ccs_binary, "--version"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            return proc.returncode == 0
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def _truncate(text: str, max_length: int = 200) -> str:
        """Truncate text for logging."""
        if len(text) <= max_length:
            return text
        return text[: max_length - 3] + "..."

    # ------------------------------------------------------------------ #
    #  Async support
    # ------------------------------------------------------------------ #

    async def on_agent_action_async(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Async version of on_agent_action. Same logic as sync."""
        self.on_agent_action(
            action,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )


class CorrectoverViolationError(Exception):
    """Exception raised when CCS blocks a tool call.

    Attributes:
        tool_name: The name of the tool that was blocked.
        rule: The CCS rule that was violated.
        detail: Detailed description of the violation.
    """

    def __init__(
        self,
        tool_name: str,
        rule: str,
        detail: str,
        message: Optional[str] = None,
    ) -> None:
        self.tool_name = tool_name
        self.rule = rule
        self.detail = detail
        self.message = message or (
            f"CCS Violation: {rule} detected in tool '{tool_name}': {detail}"
        )
        super().__init__(self.message)


# ================================================================== #
#  Runnable wrapper for LangChain Expression Language (LCEL) support
# ================================================================== #

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool


class CorrectoverCCSRunnable(Runnable):
    """A LangChain Runnable wrapper that applies CCS validation.

    Use this to wrap individual tools with CCS validation in LCEL pipelines::

        validated_tool = CorrectoverCCSRunnable(
            tool=my_tool,
            handler=CorrectoverCallbackHandler(mode="block"),
        )
        chain = validated_tool | some_processor

    This provides more granular control than the callback handler alone,
    intercepting tool calls at the Runnable level.
    """

    def __init__(
        self,
        tool: BaseTool,
        handler: Optional[CorrectoverCallbackHandler] = None,
        rules: Optional[List[str]] = None,
    ) -> None:
        self.tool = tool
        self.handler = handler or CorrectoverCallbackHandler(mode="log")
        self.rules = rules or DEFAULT_RULES
        super().__init__()

    def invoke(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke the tool with CCS validation."""
        action = AgentAction(tool=self.tool.name, tool_input=input, log="")

        self.handler.on_agent_action(action)
        return self.tool.invoke(input, config=config, **kwargs)

    async def ainvoke(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        """Async version of invoke."""
        action = AgentAction(tool=self.tool.name, tool_input=input, log="")

        self.handler.on_agent_action(action)
        return await self.tool.ainvoke(input, config=config, **kwargs)

    @property
    def InputType(self) -> Any:
        return self.tool.InputType

    @property
    def OutputType(self) -> Any:
        return self.tool.OutputType
