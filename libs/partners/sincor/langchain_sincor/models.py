"""Data models for the SINCOR platform API."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    """Lifecycle status of a SINCOR task."""

    OPEN = "open"
    AUCTION_OPEN = "auction_open"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    SETTLED = "settled"
    RELEASED = "released"
    REFUNDED = "refunded"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class TaskKind(str, Enum):
    """Kind of SINCOR task."""

    ESCROW = "escrow"
    AUCTION = "auction"  # CNP (Contract Net Protocol) auction


@dataclass
class Bid:
    """A bid placed on a SINCOR auction task."""

    id: str
    task_id: str
    bidder_id: str
    amount: float
    currency: str = "USDC"
    message: str = ""
    eta_seconds: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, d: dict) -> Bid:
        """Build a `Bid` from a platform API payload."""
        return cls(
            id=str(d.get("id") or d.get("bid_id")),
            task_id=str(d.get("task_id", "")),
            bidder_id=str(d.get("bidder_id") or d.get("agent_id", "")),
            amount=float(d.get("amount", 0)),
            currency=str(d.get("currency", "USDC")),
            message=str(d.get("message", "")),
            eta_seconds=d.get("eta_seconds"),
            raw=dict(d),
        )


@dataclass
class EscrowTask:
    """A SINCOR task (escrowed or auction)."""

    id: str
    title: str
    description: str = ""
    budget_amount: float = 0.0
    currency: str = "USDC"
    status: TaskStatus = TaskStatus.OPEN
    kind: TaskKind = TaskKind.ESCROW
    accepted_bid_id: str | None = None
    idempotency_key: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, d: dict) -> EscrowTask:
        """Build an `EscrowTask` from a platform API payload."""
        try:
            status = TaskStatus(d.get("status", "open"))
        except ValueError:
            status = TaskStatus.OPEN
        try:
            kind = TaskKind(d.get("kind", "escrow"))
        except ValueError:
            kind = TaskKind.ESCROW
        return cls(
            id=str(d.get("id") or d.get("task_id")),
            title=str(d.get("title", "")),
            description=str(d.get("description", "")),
            budget_amount=float(d.get("budget_amount") or d.get("amount") or 0),
            currency=str(d.get("currency", "USDC")),
            status=status,
            kind=kind,
            accepted_bid_id=d.get("accepted_bid_id"),
            idempotency_key=d.get("idempotency_key"),
            raw=dict(d),
        )


@dataclass
class SettlementRecord:
    """Settlement record for a task — the platform fee is explicit, never hidden."""

    task_id: str
    bid_id: str | None
    gross_amount: float
    platform_fee: float
    platform_fee_rate: float  # 0.05 = the 5% platform fee
    net_to_worker: float
    currency: str = "USDC"
    status: str = "settled"
    tx_hash: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, d: dict) -> SettlementRecord:
        """Build a `SettlementRecord` from a platform API payload."""
        gross = float(d.get("gross_amount") or d.get("amount") or 0)
        fee = float(d.get("platform_fee", round(gross * 0.05, 6)))
        net = float(d.get("net_to_worker", round(gross - fee, 6)))
        return cls(
            task_id=str(d.get("task_id", "")),
            bid_id=d.get("bid_id"),
            gross_amount=gross,
            platform_fee=fee,
            platform_fee_rate=float(d.get("platform_fee_rate", 0.05)),
            net_to_worker=net,
            currency=str(d.get("currency", "USDC")),
            status=str(d.get("status", "settled")),
            tx_hash=d.get("tx_hash"),
            raw=dict(d),
        )
