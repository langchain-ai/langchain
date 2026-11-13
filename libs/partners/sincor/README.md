# langchain-sincor

LangChain tools for [SINCOR](https://getsincor.com): escrowed tasks and
CNP (Contract Net Protocol) auctions with a settlement record that makes
the 5% platform fee explicit — instead of a single pay-and-pray call.

> **What is live vs. pilot — read this first.**
> CNP auctions, escrowed task creation, pool settlement (5% platform
> fee), and KYA telemetry are **live on https://getsincor.com** today,
> and the AP2 discovery document is served at
> `https://getsincor.com/.well-known/ap2.json`.
> SEAL (EIP-712 settlement authorization) and SIGIL (state-commitment
> verification) are **integration-pilot framing** — designed interfaces,
> not settled mainnet primitives. **No SINCOR settlement contract is
> deployed on Base mainnet**, and nothing in this package claims
> otherwise.
> The SINCOR token is **platform access, not an investment**.

## Installation

```bash
pip install langchain-sincor
```

The verified-payment companion for x402 endpoints is published on PyPI
as [`sincor-x402`](https://pypi.org/project/sincor-x402/) and is
available as an extra:

```bash
pip install "langchain-sincor[x402]"   # pins sincor-x402==0.1.0
```

## Usage

```python
from langchain_sincor import SincorAuctionTool, SincorEscrowTool

escrow = SincorEscrowTool()
auction = SincorAuctionTool()

import json

out = json.loads(
    auction.invoke({"action": "post", "title": "Summarise this dataset",
                    "budget_amount": 100.0, "currency": "USDC"})
)
task_id = out["task_id"]
bids = json.loads(auction.invoke({"action": "list_bids", "task_id": task_id}))["bids"]
auction.invoke({"action": "accept", "task_id": task_id, "bid_id": bids[0]["bid_id"]})
settled = json.loads(auction.invoke({"action": "settle", "task_id": task_id}))
assert settled["platform_fee"] == round(settled["gross_amount"] * 0.05, 6)
```

Why this shape, for LangGraph graphs in particular:

- **Idempotency** — `create` accepts an `idempotency_key`; replaying a
  graph node returns the same task, not a second escrow.
- **Audit trail** — every call returns stable `task_id` / `bid_id` ids
  and, on settle, a settlement record that can be persisted in graph
  state / checkpoints.
- **Verifier hook** — funds move in two explicit steps (`accept`, then
  `settle` / `release`); the settlement record exposes gross, the 5%
  platform fee, and net to worker for a verifier node to assert on.

## Tests

Unit tests run entirely against an in-memory mock transport (no
network): full loop post → bid → accept → settle with the fee asserted,
plus the escrow idempotency path.

```bash
make test
```
