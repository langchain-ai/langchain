"""Noma guardrail middleware for agents."""

import asyncio
import json
import logging
import os
import threading
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse

# Use stdlib urlopen to avoid adding an external HTTP dependency.
from urllib.request import Request, urlopen

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, messages_to_dict
from langchain_core.runnables import run_in_executor
from typing_extensions import override

from langchain.agents.middleware.types import AgentMiddleware, AgentState, hook_config

if TYPE_CHECKING:
    from langgraph.runtime import Runtime


_ANONYMIZABLE_DETECTORS = {"sensitiveData", "dataDetector"}
_BLOCK_MESSAGE_PREFIX = "Request blocked"
_HTTP_SUCCESS_LIMIT = 300
_ALLOWED_URL_SCHEMES = {"http", "https"}
logger = logging.getLogger(__name__)


class _NomaAPIError(RuntimeError):
    """Raised when the Noma API request fails."""


class _NomaClient:
    ENDPOINT = "/ai-dr/v2/langchain/guardrails/scan"
    DEFAULT_API_BASE = "https://api.noma.security"
    DEFAULT_TOKEN_URL = "https://api.noma.security/auth"  # noqa: S105
    DEFAULT_TIMEOUT = 60.0

    def __init__(
        self,
        *,
        api_base: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        token_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.api_base = (
            api_base or os.environ.get("NOMA_API_BASE") or self.DEFAULT_API_BASE
        ).rstrip("/")
        self.client_id = client_id or os.environ.get("NOMA_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("NOMA_CLIENT_SECRET")
        self.token_url = token_url or os.environ.get("NOMA_TOKEN_URL") or self.DEFAULT_TOKEN_URL
        self.timeout = timeout or self.DEFAULT_TIMEOUT

        if not self.client_id or not self.client_secret:
            msg = "client_id and client_secret are required for Noma OAuth"
            raise ValueError(msg)

    def _post_json(
        self, url: str, payload: dict[str, Any], headers: dict[str, str]
    ) -> dict[str, Any]:
        parsed_url = urlparse(url)
        if parsed_url.scheme not in _ALLOWED_URL_SCHEMES:
            raise _NomaAPIError

        body = json.dumps(payload).encode("utf-8")
        request = Request(url, data=body, method="POST", headers=headers)  # noqa: S310
        try:
            with urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                status = response.status
                data = response.read().decode("utf-8")
        except HTTPError as exc:
            raise _NomaAPIError from exc
        except URLError as exc:
            raise _NomaAPIError from exc

        if status >= _HTTP_SUCCESS_LIMIT:
            raise _NomaAPIError

        return json.loads(data) if data else {}

    def _fetch_token(self) -> str:
        payload = {"clientId": self.client_id, "secret": self.client_secret}
        response = self._post_json(
            self.token_url,
            payload,
            {"Content-Type": "application/json"},
        )
        token = response.get("accessToken")
        if not token:
            msg = "OAuth response missing accessToken"
            raise RuntimeError(msg)
        return token

    def scan(self, payload: dict[str, Any]) -> dict[str, Any]:
        token = self._fetch_token()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        url = urljoin(f"{self.api_base}/", self.ENDPOINT.lstrip("/"))
        return self._post_json(url, payload, headers)

    async def ascan(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await run_in_executor(None, self.scan, payload)


class NomaGuardrailMiddleware(AgentMiddleware):
    """Noma guardrail middleware for LangChain agents."""

    def __init__(
        self,
        *,
        client_id: str | None = None,
        client_secret: str | None = None,
        token_url: str | None = None,
        api_base: str | None = None,
        application_id: str | None = None,
        monitor_mode: bool = False,
        block_failures: bool = False,
    ) -> None:
        """Initialize the Noma guardrail middleware.

        Args:
            client_id: Noma client ID. Defaults to env NOMA_CLIENT_ID.
            client_secret: Noma client secret. Defaults to env NOMA_CLIENT_SECRET.
            token_url: Authentication endpoint. Defaults to env NOMA_TOKEN_URL or
                https://api.noma.security/auth.
            api_base: Noma API base URL. Defaults to env NOMA_API_BASE or
                https://api.noma.security.
            application_id: Application identifier for Noma context. Defaults to
                env NOMA_APPLICATION_ID or "langchain".
            monitor_mode: If True, run checks in the background and do not block
                or anonymize. Defaults to False.
            block_failures: If True, block on Noma errors. Defaults to False (fail-open).
        """
        super().__init__()
        self._client = _NomaClient(
            api_base=api_base,
            client_id=client_id,
            client_secret=client_secret,
            token_url=token_url,
        )
        self.application_id = application_id or os.environ.get("NOMA_APPLICATION_ID") or "langchain"
        self.monitor_mode = monitor_mode
        self.block_failures = block_failures
        self._monitor_tasks: set[asyncio.Task[None]] = set()

    def _build_default_context(self, runtime: "Runtime") -> dict[str, Any]:
        context: dict[str, Any] = {"applicationId": self.application_id}

        runtime_context = getattr(runtime, "context", None)
        if isinstance(runtime_context, dict):
            session_id = runtime_context.get("sessionId") or runtime_context.get("session_id")
            request_id = runtime_context.get("requestId") or runtime_context.get("request_id")
            labels = runtime_context.get("labels")
            user_id = runtime_context.get("userId") or runtime_context.get("user_id")
            ip_address = runtime_context.get("ipAddress") or runtime_context.get("ip_address")
            if session_id:
                context["sessionId"] = session_id
            if request_id:
                context["requestId"] = request_id
            if labels:
                context["labels"] = labels
            if user_id:
                context["userId"] = user_id
            if ip_address:
                context["ipAddress"] = ip_address
        elif runtime_context:
            session_id = getattr(runtime_context, "session_id", None)
            request_id = getattr(runtime_context, "request_id", None)
            labels = getattr(runtime_context, "labels", None)
            user_id = getattr(runtime_context, "user_id", None)
            ip_address = getattr(runtime_context, "ip_address", None)
            if session_id:
                context["sessionId"] = session_id
            if request_id:
                context["requestId"] = request_id
            if labels:
                context["labels"] = labels
            if user_id:
                context["userId"] = user_id
            if ip_address:
                context["ipAddress"] = ip_address

        return context

    def _serialize_messages(
        self, messages: Sequence[AnyMessage | dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if messages and isinstance(messages[0], dict):
            return list(cast("Sequence[dict[str, Any]]", messages))
        return messages_to_dict(cast("Sequence[AnyMessage]", messages))

    def _build_payload(
        self, state: AgentState[Any], runtime: "Runtime", phase: str
    ) -> dict[str, Any]:
        messages = state.get("messages", [])
        payload: dict[str, Any] = {
            "messages": self._serialize_messages(messages),
            "phase": phase,
        }

        if "structured_response" in state:
            payload["structured_response"] = state["structured_response"]

        context = self._build_default_context(runtime)
        if context is not None:
            payload["x-noma-context"] = context

        return payload

    @staticmethod
    def _is_positive_result(result_obj: Any) -> bool:
        return isinstance(result_obj, dict) and result_obj.get("result") is True

    def _only_anonymizable_detectors_triggered(self, results: dict[str, Any]) -> bool:
        has_sensitive = False
        has_blocking = False

        for key, value in results.items():
            if key in {"anonymizedContent", "metadata"}:
                continue
            if not isinstance(value, dict):
                continue

            if key in _ANONYMIZABLE_DETECTORS:
                for detector_result in value.values():
                    if self._is_positive_result(detector_result):
                        has_sensitive = True
            elif value.get("result") is not None:
                if self._is_positive_result(value):
                    has_blocking = True
            else:
                for nested_result in value.values():
                    if self._is_positive_result(nested_result):
                        has_blocking = True

        return has_sensitive and not has_blocking

    def _extract_anonymized_content(self, response: dict[str, Any], role: str) -> str | None:
        scan_result = response.get("scanResult", [])
        if not isinstance(scan_result, list):
            return None
        for item in scan_result:
            if not isinstance(item, dict) or item.get("role") != role:
                continue
            results = item.get("results", {})
            if not isinstance(results, dict):
                continue
            anonymized = results.get("anonymizedContent")
            if isinstance(anonymized, dict):
                return anonymized.get("anonymized")
        return None

    def _should_anonymize(self, response: dict[str, Any], role: str) -> bool:
        if self.monitor_mode:
            return False

        scan_result = response.get("scanResult", [])
        if not isinstance(scan_result, list):
            return False
        for item in scan_result:
            if not isinstance(item, dict) or item.get("role") != role:
                continue
            results = item.get("results", {})
            if isinstance(results, dict):
                return self._only_anonymizable_detectors_triggered(results)
        return False

    def _apply_anonymization(
        self,
        messages: list[AnyMessage],
        role: str,
        anonymized_content: str,
    ) -> dict[str, Any] | None:
        if not anonymized_content:
            return None

        new_messages = list(messages)
        for i in range(len(new_messages) - 1, -1, -1):
            msg = new_messages[i]
            if role == "user" and isinstance(msg, HumanMessage):
                new_messages[i] = HumanMessage(
                    content=anonymized_content,
                    id=msg.id,
                    name=msg.name,
                )
                return {"messages": new_messages}
            if role == "assistant" and isinstance(msg, AIMessage):
                new_messages[i] = AIMessage(
                    content=anonymized_content,
                    id=msg.id,
                    name=msg.name,
                    tool_calls=msg.tool_calls,
                )
                return {"messages": new_messages}

        return None

    def _build_block_response(self, reason: str) -> dict[str, Any]:
        return {
            "messages": [AIMessage(content=f"Request blocked by Noma guardrail: {reason}")],
            "jump_to": "end",
        }

    def _handle_api_failure(self, phase: str) -> dict[str, Any] | None:
        if self.monitor_mode or not self.block_failures:
            return None
        return {
            "messages": [AIMessage(content=f"Request blocked: Noma guardrail failed ({phase})")],
            "jump_to": "end",
        }

    def _is_block_message(self, message: AnyMessage) -> bool:
        return (
            isinstance(message, AIMessage)
            and isinstance(message.content, str)
            and message.content.startswith(_BLOCK_MESSAGE_PREFIX)
        )

    def _last_message_is_block(self, messages: list[AnyMessage]) -> bool:
        if not messages:
            return False
        return self._is_block_message(messages[-1])

    def _handle_scan_response(
        self,
        response: dict[str, Any],
        role: str,
        messages: list[AnyMessage],
    ) -> dict[str, Any] | None:
        if self.monitor_mode:
            return None

        if not response.get("aggregatedScanResult"):
            return None

        if self._should_anonymize(response, role):
            anonymized = self._extract_anonymized_content(response, role)
            if anonymized:
                return self._apply_anonymization(messages, role, anonymized)

        if self._last_message_is_block(messages):
            return {"jump_to": "end"}

        return self._build_block_response("unsafe content detected")

    def _log_monitor_result(self, phase: str, role: str, response: dict[str, Any] | None) -> None:
        if not isinstance(response, dict):
            logger.info("Noma monitor scan result: phase=%s role=%s response=invalid", phase, role)
            return

        aggregated = response.get("aggregatedScanResult")
        scan_result = response.get("scanResult", [])
        scan_items = scan_result if isinstance(scan_result, list) else []
        roles = [
            item.get("role")
            for item in scan_items
            if isinstance(item, dict) and isinstance(item.get("role"), str)
        ]
        logger.info(
            "Noma monitor scan result: phase=%s role=%s aggregated=%s scan_items=%d roles=%s",
            phase,
            role,
            aggregated,
            len(scan_items),
            roles,
        )

    def _run_scan(
        self,
        state: AgentState[Any],
        runtime: "Runtime",
        *,
        phase: str,
        role: str,
        scan: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        if not messages:
            return None

        if phase == "after_agent" and self._last_message_is_block(messages):
            return None

        payload = self._build_payload(state, runtime, phase)
        if self.monitor_mode:

            def run_monitor_scan() -> None:
                try:
                    response = scan(payload)
                except Exception as exc:
                    logger.warning(
                        "Noma monitor scan failed: phase=%s role=%s error=%s",
                        phase,
                        role,
                        exc,
                    )
                    return
                self._log_monitor_result(phase, role, response)

            threading.Thread(target=run_monitor_scan, daemon=True).start()
            return None

        try:
            response = scan(payload)
        except Exception:
            if self._last_message_is_block(messages):
                return {"jump_to": "end"}
            return self._handle_api_failure(phase)

        return self._handle_scan_response(response, role, messages)

    async def _arun_scan(
        self,
        state: AgentState[Any],
        runtime: "Runtime",
        *,
        phase: str,
        role: str,
        scan: Callable[[dict[str, Any]], Any],
    ) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        if not messages:
            return None

        if phase == "after_agent" and self._last_message_is_block(messages):
            return None

        payload = self._build_payload(state, runtime, phase)
        if self.monitor_mode:

            async def run_monitor_scan() -> None:
                try:
                    response = await scan(payload)
                except Exception as exc:
                    logger.warning(
                        "Noma monitor scan failed: phase=%s role=%s error=%s",
                        phase,
                        role,
                        exc,
                    )
                    return
                self._log_monitor_result(phase, role, response)

            task = asyncio.create_task(run_monitor_scan())
            self._monitor_tasks.add(task)
            task.add_done_callback(self._monitor_tasks.discard)
            return None

        try:
            response = await scan(payload)
        except Exception:
            if self._last_message_is_block(messages):
                return {"jump_to": "end"}
            return self._handle_api_failure(phase)

        return self._handle_scan_response(response, role, messages)

    @hook_config(can_jump_to=["end"])
    @override
    def before_agent(self, state: AgentState[Any], runtime: "Runtime") -> dict[str, Any] | None:
        """Scan user messages before agent execution."""
        return self._run_scan(
            state,
            runtime,
            phase="before_agent",
            role="user",
            scan=self._client.scan,
        )

    @hook_config(can_jump_to=["end"])
    async def abefore_agent(
        self, state: AgentState[Any], runtime: "Runtime"
    ) -> dict[str, Any] | None:
        """Async scan user messages before agent execution."""
        return await self._arun_scan(
            state,
            runtime,
            phase="before_agent",
            role="user",
            scan=self._client.ascan,
        )

    @hook_config(can_jump_to=["end"])
    @override
    def after_agent(self, state: AgentState[Any], runtime: "Runtime") -> dict[str, Any] | None:
        """Scan assistant messages after agent execution."""
        return self._run_scan(
            state,
            runtime,
            phase="after_agent",
            role="assistant",
            scan=self._client.scan,
        )

    @hook_config(can_jump_to=["end"])
    async def aafter_agent(
        self, state: AgentState[Any], runtime: "Runtime"
    ) -> dict[str, Any] | None:
        """Async scan assistant messages after agent execution."""
        return await self._arun_scan(
            state,
            runtime,
            phase="after_agent",
            role="assistant",
            scan=self._client.ascan,
        )
