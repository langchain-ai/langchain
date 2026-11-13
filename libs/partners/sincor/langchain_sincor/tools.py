"""Tools for the SINCOR escrow and CNP auction platform."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, model_validator

from langchain_sincor.client import SincorClient


class EscrowInput(BaseModel):
    """Input schema for `SincorEscrowTool`."""

    action: str = Field(
        description="One of: create, status, release, refund, settlement"
    )
    title: str = Field(default="", description="Task title (create only)")
    description: str = Field(default="", description="Task description (create only)")
    budget_amount: float = Field(default=0.0, description="Escrow budget, e.g. in USDC")
    currency: str = Field(default="USDC")
    task_id: str = Field(
        default="", description="Task id (status/release/refund/settlement)"
    )
    idempotency_key: str | None = Field(
        default=None,
        description="Idempotency key for create — the same key returns the same task",
    )


class AuctionInput(BaseModel):
    """Input schema for `SincorAuctionTool`."""

    action: str = Field(description="One of: post, list_bids, accept, settle, status")
    title: str = Field(default="", description="Task title (post only)")
    description: str = Field(default="", description="Task description (post only)")
    budget_amount: float = Field(
        default=0.0, description="Auction budget ceiling, e.g. in USDC"
    )
    currency: str = Field(default="USDC")
    task_id: str = Field(
        default="", description="Task id (list_bids/accept/settle/status)"
    )
    bid_id: str = Field(default="", description="Bid id (accept only)")


class SincorEscrowTool(BaseTool):  # type: ignore[override]
    """Create and manage escrowed SINCOR tasks.

    Setup:
        Install `langchain-sincor`:

        ``bash
        pip install -U langchain-sincor
        ``

    Instantiation:
        ``python
        from langchain_sincor import SincorEscrowTool

        tool = SincorEscrowTool()
        ``

    Invocation with args:
        ``python
        tool.invoke(
            {
                "action": "create",
                "title": "Summarise this dataset",
                "budget_amount": 50.0,
                "idempotency_key": "graph-run-123",
            }
        )
        ``

        ``python
        '{"task_id": "tsk_...", "status": "open", "kind": "escrow",
          "budget_amount": 50.0, "currency": "USDC"}'
        ``

    Funds stay in escrow until an explicit `release` (which returns the
    settlement record, including the 5% platform fee and the net to the
    worker) or `refund`. Replaying a create with the same
    `idempotency_key` returns the same task, so a re-run graph node does
    not open a second escrow.
    """

    name: str = "sincor_escrow"
    description: str = (
        "Create an escrowed SINCOR task, check its status, release escrow "
        "(the settlement record shows the 5% platform fee and net to worker), "
        "or refund it. Use when payment must not be pay-and-pray: funds are "
        "escrowed until release."
    )
    args_schema: type[BaseModel] = EscrowInput
    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: SincorClient | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _default_client(self) -> SincorEscrowTool:
        if self.client is None:
            self.client = SincorClient()
        return self

    def _run(
        self,
        action: str = "create",
        title: str = "",
        description: str = "",
        budget_amount: float = 0.0,
        currency: str = "USDC",
        task_id: str = "",
        idempotency_key: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> str:
        c = self.client
        if c is None:  # pragma: no cover - the validator guarantees a client
            msg = "SincorEscrowTool has no client configured."
            raise ValueError(msg)
        if action == "create":
            t = c.create_escrowed_task(
                title,
                description,
                budget_amount,
                currency,
                idempotency_key=idempotency_key,
            )
            return json.dumps(
                {
                    "task_id": t.id,
                    "status": t.status.value,
                    "kind": t.kind.value,
                    "budget_amount": t.budget_amount,
                    "currency": t.currency,
                }
            )
        if action == "status":
            t = c.check_status(task_id)
            return json.dumps(
                {
                    "task_id": t.id,
                    "status": t.status.value,
                    "accepted_bid_id": t.accepted_bid_id,
                }
            )
        if action == "release":
            s = c.release(task_id)
            return json.dumps(
                {
                    "task_id": s.task_id,
                    "status": s.status,
                    "gross_amount": s.gross_amount,
                    "platform_fee": s.platform_fee,
                    "platform_fee_rate": s.platform_fee_rate,
                    "net_to_worker": s.net_to_worker,
                    "currency": s.currency,
                }
            )
        if action == "refund":
            t = c.refund(task_id)
            return json.dumps({"task_id": t.id, "status": t.status.value})
        if action == "settlement":
            s = c.get_settlement(task_id)
            return json.dumps(
                {
                    "task_id": s.task_id,
                    "gross_amount": s.gross_amount,
                    "platform_fee": s.platform_fee,
                    "net_to_worker": s.net_to_worker,
                }
            )
        msg = f"unknown action {action!r}"
        raise ValueError(msg)

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        return self._run(*args, **kwargs)


class SincorAuctionTool(BaseTool):  # type: ignore[override]
    """Post CNP auction tasks on SINCOR and settle them.

    Setup:
        Install `langchain-sincor`:

        ``bash
        pip install -U langchain-sincor
        ``

    Instantiation:
        ``python
        from langchain_sincor import SincorAuctionTool

        tool = SincorAuctionTool()
        ``

    Invocation with args:
        ``python
        tool.invoke({"action": "post", "title": "Data pull", "budget_amount": 100.0})
        tool.invoke({"action": "list_bids", "task_id": task_id})
        tool.invoke({"action": "accept", "task_id": task_id, "bid_id": bid_id})
        settled = tool.invoke({"action": "settle", "task_id": task_id})
        ``

    Settlement is a two-step accept -> settle flow: no funds move on a
    single pay call, and the settlement record exposes the gross amount,
    the 5% platform fee, and the net to the worker, so a verifier node in
    a LangGraph graph can assert on the record before proceeding.
    """

    name: str = "sincor_auction"
    description: str = (
        "Post a CNP auction task on SINCOR, list bids, accept a bid, and settle. "
        "Settlement shows gross, the 5% platform fee, and net to worker. "
        "Use for competitive agent-to-agent task bidding instead of a single "
        "pay-and-pray call."
    )
    args_schema: type[BaseModel] = AuctionInput
    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: SincorClient | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _default_client(self) -> SincorAuctionTool:
        if self.client is None:
            self.client = SincorClient()
        return self

    def _run(
        self,
        action: str = "post",
        title: str = "",
        description: str = "",
        budget_amount: float = 0.0,
        currency: str = "USDC",
        task_id: str = "",
        bid_id: str = "",
        run_manager: CallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> str:
        c = self.client
        if c is None:  # pragma: no cover - the validator guarantees a client
            msg = "SincorAuctionTool has no client configured."
            raise ValueError(msg)
        if action == "post":
            t = c.post_auction_task(title, description, budget_amount, currency)
            return json.dumps(
                {
                    "task_id": t.id,
                    "status": t.status.value,
                    "kind": t.kind.value,
                    "budget_amount": t.budget_amount,
                }
            )
        if action == "list_bids":
            bids = c.list_bids(task_id)
            return json.dumps(
                {
                    "task_id": task_id,
                    "bids": [
                        {
                            "bid_id": b.id,
                            "bidder_id": b.bidder_id,
                            "amount": b.amount,
                            "currency": b.currency,
                        }
                        for b in bids
                    ],
                }
            )
        if action == "accept":
            t = c.accept_bid(task_id, bid_id)
            return json.dumps(
                {
                    "task_id": t.id,
                    "status": t.status.value,
                    "accepted_bid_id": t.accepted_bid_id,
                }
            )
        if action in ("settle", "release"):
            s = c.settle(task_id)
            return json.dumps(
                {
                    "task_id": s.task_id,
                    "bid_id": s.bid_id,
                    "status": s.status,
                    "gross_amount": s.gross_amount,
                    "platform_fee": s.platform_fee,
                    "platform_fee_rate": s.platform_fee_rate,
                    "net_to_worker": s.net_to_worker,
                    "currency": s.currency,
                }
            )
        if action == "status":
            t = c.check_status(task_id)
            return json.dumps(
                {
                    "task_id": t.id,
                    "status": t.status.value,
                    "accepted_bid_id": t.accepted_bid_id,
                }
            )
        msg = f"unknown action {action!r}"
        raise ValueError(msg)

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        return self._run(*args, **kwargs)
