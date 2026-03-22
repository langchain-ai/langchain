"""SafetyMD LangChain Tool — checks payment addresses for risk before sending funds."""

from __future__ import annotations

import json
from typing import Any, Optional, Type

import requests
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

SAFETYMD_API_BASE = "https://safetymd.p-u-c.workers.dev/v1/check"
DEFAULT_CHAIN = "base"
DEFAULT_TIMEOUT = 3  # seconds


class SafetyMDInput(BaseModel):
    """Input schema for SafetyMDTool."""

    address: str = Field(
        ...,
        description="The payment address to check (e.g. 0x1234...).",
    )
    chain: str = Field(
        default=DEFAULT_CHAIN,
        description="Chain to check on (e.g. 'base', 'ethereum', 'arbitrum'). Defaults to 'base'.",
    )


def _format_report(address: str, chain: str, data: dict[str, Any], checks_remaining: Optional[str]) -> str:
    """Format a safety check result into a human-readable string an LLM can reason over."""
    safe: bool = data.get("safe", False)
    risk: str = data.get("risk", "unknown")
    reason: str = data.get("reason", "No reason provided.")
    service: dict = data.get("service", {})
    signals: dict = data.get("signals", {})

    status_emoji = "✅" if safe else "🚨"
    risk_upper = risk.upper()

    lines = [
        f"{status_emoji} SafetyMD Address Check",
        f"Address : {address}",
        f"Chain   : {chain}",
        f"Safe    : {'YES' if safe else 'NO'}",
        f"Risk    : {risk_upper}",
        f"Reason  : {reason}",
    ]

    if service:
        lines.append(f"Service : {json.dumps(service)}")

    if signals:
        lines.append("Signals :")
        for k, v in signals.items():
            lines.append(f"  {k}: {v}")

    if checks_remaining is not None:
        lines.append(f"Free checks remaining today: {checks_remaining}")

    lines.append("")
    lines.append("--- raw JSON ---")
    lines.append(json.dumps(data, indent=2))

    return "\n".join(lines)


class SafetyMDTool(BaseTool):
    """LangChain tool that checks whether a payment address is safe before funds are sent.

    Uses the SafetyMD API (https://safetymd.p-u-c.workers.dev) to assess risk.
    Free tier: 10 checks/day, no API key required.
    """

    name: str = "check_payment_address"
    description: str = (
        "Check if a payment address is safe before sending funds. "
        "ALWAYS use this tool before any payment or transfer operation. "
        "Input: JSON with 'address' (required, e.g. '0x...') and optional 'chain' "
        "(default: 'base'). Supported chains: base, ethereum, arbitrum, polygon, etc. "
        "Returns a safety report with risk level (low/medium/high/critical) and reason."
    )
    args_schema: Type[BaseModel] = SafetyMDInput

    def _run(self, address: str, chain: str = DEFAULT_CHAIN) -> str:  # type: ignore[override]
        """Synchronous safety check."""
        return _check_address(address=address, chain=chain)

    async def _arun(self, address: str, chain: str = DEFAULT_CHAIN) -> str:  # type: ignore[override]
        """Async safety check — runs blocking HTTP call in thread pool."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: _check_address(address=address, chain=chain))


def _check_address(address: str, chain: str = DEFAULT_CHAIN) -> str:
    """Core HTTP call to SafetyMD API. Never raises — always returns a string."""
    address = address.strip()
    chain = chain.strip() or DEFAULT_CHAIN
    url = f"{SAFETYMD_API_BASE}/{address}"
    params = {"chain": chain}

    try:
        response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    except requests.exceptions.Timeout:
        return (
            f"⚠️ SafetyMD check timed out after {DEFAULT_TIMEOUT}s for address {address} on {chain}. "
            "Treat as UNKNOWN risk — verify manually before sending funds."
        )
    except requests.exceptions.ConnectionError as exc:
        return (
            f"⚠️ SafetyMD connection error for address {address}: {exc}. "
            "Cannot verify safety — do NOT send funds without manual verification."
        )
    except requests.exceptions.RequestException as exc:
        return (
            f"⚠️ SafetyMD request error for address {address}: {exc}. "
            "Cannot verify safety."
        )

    checks_remaining = response.headers.get("x-free-checks-remaining")

    if response.status_code == 429:
        return (
            f"⚠️ SafetyMD rate limit exceeded (429) for address {address}. "
            "Free daily quota exhausted. Cannot verify safety — consider upgrading or retrying tomorrow."
        )

    if response.status_code != 200:
        return (
            f"⚠️ SafetyMD API returned HTTP {response.status_code} for address {address}. "
            f"Response: {response.text[:200]}. Cannot verify safety."
        )

    try:
        data: dict[str, Any] = response.json()
    except ValueError:
        return (
            f"⚠️ SafetyMD returned non-JSON response for address {address}: {response.text[:200]}."
        )

    return _format_report(address=address, chain=chain, data=data, checks_remaining=checks_remaining)
