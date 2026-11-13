"""Typed client for the SINCOR platform API (standard library only).

Live today on https://getsincor.com: CNP auction tasks, escrowed task
creation, and pool settlement with a 5% platform fee. SEAL (EIP-712
settlement authorization) and SIGIL (state-commitment verification) are
integration-pilot surfaces; this client does not claim them as settled
mainnet primitives, and no settlement contract is deployed on Base
mainnet. The SINCOR token is platform access, not an investment.

The companion verified-payment package for x402 endpoints is published
separately as `sincor-x402` on PyPI and is an optional dependency of
`langchain-sincor` (the `x402` extra).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Protocol, runtime_checkable

from langchain_sincor.models import Bid, EscrowTask, SettlementRecord


class SincorError(Exception):
    """Base error for the SINCOR client."""


class SincorAPIError(SincorError):
    """Error returned by (or while calling) the SINCOR platform API."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        """Initialize with a message and an optional HTTP status code."""
        super().__init__(message)
        self.status_code = status_code


class SincorNotFound(SincorAPIError):
    """The requested task or bid does not exist."""


@runtime_checkable
class Transport(Protocol):
    """Transport protocol: one JSON request/response round trip."""

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
        params: dict | None = None,
    ) -> dict:
        """Perform a request against the platform API and return parsed JSON."""
        ...


class HttpTransport:
    """HTTP transport based on `urllib` (no third-party dependencies)."""

    def __init__(
        self,
        base_url: str = "https://getsincor.com",
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the transport.

        Args:
            base_url: Base URL of the SINCOR platform.
            api_key: Optional bearer token.
            timeout: Per-request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
        params: dict | None = None,
    ) -> dict:
        """Perform an HTTP request and return the parsed JSON body."""
        url = f"{self.base_url}{path}"
        if params:
            qs = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{qs}"
        data = json.dumps(json_body).encode() if json_body is not None else None
        req = urllib.request.Request(url, data=data, method=method.upper())  # noqa: S310
        req.add_header("User-Agent", "Mozilla/5.0 langchain-sincor/0.1.0")
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                body = resp.read().decode()
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            if e.code == 404:
                msg = f"Not found: {path}"
                raise SincorNotFound(msg, status_code=404) from e
            msg = f"HTTP {e.code} on {path}"
            raise SincorAPIError(msg, status_code=e.code) from e


class SincorClient:
    """Typed client for the SINCOR platform API."""

    def __init__(
        self,
        base_url: str = "https://getsincor.com",
        api_key: str | None = None,
        transport: Transport | None = None,
    ) -> None:
        """Initialize the client.

        Args:
            base_url: Base URL of the SINCOR platform.
            api_key: Optional bearer token.
            transport: Custom transport (used by tests); defaults to
                `HttpTransport`.
        """
        self.base_url = base_url.rstrip("/")
        self.transport: Transport = (
            transport
            if transport is not None
            else HttpTransport(base_url=base_url, api_key=api_key)
        )

    # --- creation ---
    def create_escrowed_task(
        self,
        title: str,
        description: str = "",
        budget_amount: float = 0.0,
        currency: str = "USDC",
        *,
        idempotency_key: str | None = None,
    ) -> EscrowTask:
        """Create an escrowed task.

        Replaying an `idempotency_key` returns the same task instead of
        creating a second escrow.
        """
        body = {
            "title": title,
            "description": description,
            "budget_amount": budget_amount,
            "currency": currency,
            "idempotency_key": idempotency_key,
        }
        return EscrowTask.from_api(
            self.transport.request("POST", "/api/tasks", json_body=body)
        )

    def post_auction_task(
        self,
        title: str,
        description: str = "",
        budget_amount: float = 0.0,
        currency: str = "USDC",
        *,
        idempotency_key: str | None = None,
    ) -> EscrowTask:
        """Post a CNP (Contract Net Protocol) auction task.

        Workers bid, the poster accepts one bid, and settlement follows
        acceptance.
        """
        body = {
            "title": title,
            "description": description,
            "budget_amount": budget_amount,
            "currency": currency,
            "idempotency_key": idempotency_key,
        }
        return EscrowTask.from_api(
            self.transport.request("POST", "/api/tasks/auction", json_body=body)
        )

    # --- bids ---
    def list_bids(self, task_id: str) -> list[Bid]:
        """List bids on a task."""
        data = self.transport.request("GET", f"/api/tasks/{task_id}/bids")
        items = data.get("bids", data) if isinstance(data, dict) else data
        return [Bid.from_api(b) for b in items]

    def place_bid(
        self,
        task_id: str,
        amount: float,
        bidder_id: str,
        currency: str = "USDC",
        message: str = "",
    ) -> Bid:
        """Place a bid on an auction task."""
        data = self.transport.request(
            "POST",
            f"/api/tasks/{task_id}/bids",
            json_body={
                "amount": amount,
                "bidder_id": bidder_id,
                "currency": currency,
                "message": message,
            },
        )
        return Bid.from_api(data)

    def accept_bid(self, task_id: str, bid_id: str) -> EscrowTask:
        """Accept a bid, assigning the task to that bidder."""
        return EscrowTask.from_api(
            self.transport.request(
                "POST", f"/api/tasks/{task_id}/accept", json_body={"bid_id": bid_id}
            )
        )

    # --- status / settlement ---
    def check_status(self, task_id: str) -> EscrowTask:
        """Fetch the current state of a task."""
        return EscrowTask.from_api(
            self.transport.request("GET", f"/api/tasks/{task_id}")
        )

    def get_settlement(self, task_id: str) -> SettlementRecord:
        """Fetch the settlement record for a settled task."""
        return SettlementRecord.from_api(
            self.transport.request("GET", f"/api/tasks/{task_id}/settlement")
        )

    def release(self, task_id: str) -> SettlementRecord:
        """Release escrow to the accepted worker.

        Returns the settlement record with the platform fee visible.
        """
        return SettlementRecord.from_api(
            self.transport.request(
                "POST", f"/api/tasks/{task_id}/release", json_body={}
            )
        )

    def settle(self, task_id: str) -> SettlementRecord:
        """Settle a task whose bid was accepted; returns the settlement record."""
        return SettlementRecord.from_api(
            self.transport.request("POST", f"/api/tasks/{task_id}/settle", json_body={})
        )

    def refund(self, task_id: str) -> EscrowTask:
        """Refund an escrowed task to its poster."""
        return EscrowTask.from_api(
            self.transport.request("POST", f"/api/tasks/{task_id}/refund", json_body={})
        )


__all__ = [
    "HttpTransport",
    "SincorAPIError",
    "SincorClient",
    "SincorError",
    "SincorNotFound",
    "Transport",
]
