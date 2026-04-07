"""Shared HTTP client for AI Identity gateway enforcement and audit logging.

Centralises all HTTP communication with the AI Identity platform so that
``chat_models``, ``tools``, ``callback``, and ``middleware`` modules share a
single, environment-configurable implementation.

URLs default to the AI Identity cloud endpoints but can be overridden via
environment variables or constructor parameters:

* ``AI_IDENTITY_GATEWAY_URL`` — gateway enforce endpoint base
* ``AI_IDENTITY_API_URL`` — audit / management API base
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_GATEWAY_URL = "https://ai-identity-gateway.onrender.com"
_DEFAULT_API_URL = "https://ai-identity-api.onrender.com"
_DEFAULT_TIMEOUT = 5.0

_LLM_ENDPOINT = "/v1/chat/completions"


def _gateway_url(override: str | None = None) -> str:
    """Resolve the gateway base URL."""
    return override or os.environ.get("AI_IDENTITY_GATEWAY_URL", _DEFAULT_GATEWAY_URL)


def _api_url(override: str | None = None) -> str:
    """Resolve the API base URL."""
    return override or os.environ.get("AI_IDENTITY_API_URL", _DEFAULT_API_URL)


# ---------------------------------------------------------------------------
# Gateway enforcement
# ---------------------------------------------------------------------------


def enforce_access(
    api_key: str,
    agent_id: str,
    endpoint: str,
    method: str = "POST",
    *,
    fail_closed: bool = True,
    timeout: float = _DEFAULT_TIMEOUT,
    gateway_url: str | None = None,
) -> dict[str, Any]:
    """Call the AI Identity gateway to enforce access policy.

    Returns:
        Parsed JSON response from the gateway.

    Raises:
        PermissionError: If denied and *fail_closed* is ``True``.
        RuntimeError: If the gateway is unreachable and *fail_closed* is ``True``.
    """
    url = f"{_gateway_url(gateway_url)}/gateway/enforce"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                url,
                params={
                    "agent_id": agent_id,
                    "endpoint": endpoint,
                    "method": method,
                    "key_type": "runtime",
                },
                headers={"X-API-Key": api_key},
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        msg = f"[AI Identity] Gateway returned HTTP {exc.response.status_code} for {endpoint}"
        logger.error(msg)
        if fail_closed:
            raise PermissionError(msg) from exc
        warnings.warn(msg, stacklevel=3)
        return {"decision": "allow", "reason": "fail-open after gateway error"}
    except Exception as exc:
        msg = f"[AI Identity] Gateway unreachable for {endpoint}: {exc}"
        logger.error(msg)
        if fail_closed:
            raise RuntimeError(msg) from exc
        warnings.warn(msg, stacklevel=3)
        return {"decision": "allow", "reason": "fail-open after gateway error"}


async def aenforce_access(
    api_key: str,
    agent_id: str,
    endpoint: str,
    method: str = "POST",
    *,
    fail_closed: bool = True,
    timeout: float = _DEFAULT_TIMEOUT,
    gateway_url: str | None = None,
) -> dict[str, Any]:
    """Async version of :func:`enforce_access`."""
    url = f"{_gateway_url(gateway_url)}/gateway/enforce"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                url,
                params={
                    "agent_id": agent_id,
                    "endpoint": endpoint,
                    "method": method,
                    "key_type": "runtime",
                },
                headers={"X-API-Key": api_key},
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        msg = f"[AI Identity] Gateway returned HTTP {exc.response.status_code} for {endpoint}"
        logger.error(msg)
        if fail_closed:
            raise PermissionError(msg) from exc
        warnings.warn(msg, stacklevel=3)
        return {"decision": "allow", "reason": "fail-open after gateway error"}
    except Exception as exc:
        msg = f"[AI Identity] Gateway unreachable for {endpoint}: {exc}"
        logger.error(msg)
        if fail_closed:
            raise RuntimeError(msg) from exc
        warnings.warn(msg, stacklevel=3)
        return {"decision": "allow", "reason": "fail-open after gateway error"}


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------


def post_audit(
    api_key: str,
    agent_id: str,
    event_type: str,
    endpoint: str,
    decision: str,
    *,
    metadata: dict[str, Any] | None = None,
    latency_ms: float | None = None,
    fail_closed: bool = True,
    timeout: float = _DEFAULT_TIMEOUT,
    api_url: str | None = None,
) -> None:
    """Post an audit entry to the AI Identity API."""
    url = f"{_api_url(api_url)}/api/v1/audit"
    payload: dict[str, Any] = {
        "agent_id": agent_id,
        "event_type": event_type,
        "endpoint": endpoint,
        "decision": decision,
        "metadata": {**(metadata or {}), "action_type": event_type},
    }
    if latency_ms is not None:
        payload["latency_ms"] = latency_ms

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=payload, headers={"X-API-Key": api_key})
            response.raise_for_status()
    except Exception as exc:
        msg = f"[AI Identity] Audit log failed ({event_type}): {exc}"
        if fail_closed:
            raise RuntimeError(msg) from exc
        warnings.warn(msg, stacklevel=3)
        logger.warning(msg)


async def apost_audit(
    api_key: str,
    agent_id: str,
    event_type: str,
    endpoint: str,
    decision: str,
    *,
    metadata: dict[str, Any] | None = None,
    latency_ms: float | None = None,
    fail_closed: bool = True,
    timeout: float = _DEFAULT_TIMEOUT,
    api_url: str | None = None,
) -> None:
    """Async version of :func:`post_audit`."""
    url = f"{_api_url(api_url)}/api/v1/audit"
    payload: dict[str, Any] = {
        "agent_id": agent_id,
        "event_type": event_type,
        "endpoint": endpoint,
        "decision": decision,
        "metadata": {**(metadata or {}), "action_type": event_type},
    }
    if latency_ms is not None:
        payload["latency_ms"] = latency_ms

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url, json=payload, headers={"X-API-Key": api_key}
            )
            response.raise_for_status()
    except Exception as exc:
        msg = f"[AI Identity] Audit log failed ({event_type}): {exc}"
        if fail_closed:
            raise RuntimeError(msg) from exc
        warnings.warn(msg, stacklevel=3)
        logger.warning(msg)
