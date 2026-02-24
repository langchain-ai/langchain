# Secure tool execution with AECP

Agent tool calls can trigger real side effects, but most examples do not show delegated authorization, budget enforcement, or verifiable execution records. This walkthrough shows how to wrap a LangChain tool with AECP so each call is authorized before execution and recorded as a signed receipt after execution.

## Install

```bash
pip install langchain-core aecp-sdk==0.1.0 aecp-langchain==0.1.0
```

## Start AECP locally

Start a local AECP control plane with Docker Compose (from your AECP checkout):

```bash
docker compose up -d --build
```

Then, at a high level:

1. Register a tool id (for example, `tool.billing.charge`) in AECP.
2. Create a delegation that allowlists that tool and sets a small budget (for this demo, limit `1`).
3. Export the delegation token from the create-delegation response.

```bash
export AECP_BASE_URL=http://localhost:8000
export AECP_DELEGATION_TOKEN=<delegation_token>
```

## Run the example

```python
import os

from aecp_langchain import AECPToolConfig, wrap_tool
from aecp_langchain.callbacks import AECPCallbacks
from aecp_sdk import AuthorizationDenied
from langchain_core.tools import tool


@tool
def charge_customer(customer_id: str, amount_cents: int) -> dict:
    """Simulate a side-effecting charge call without external payment APIs."""
    return {
        "status": "charged",
        "customer_id": customer_id,
        "amount_cents": amount_cents,
    }


def on_authorize(permit: dict) -> None:
    print(
        "authorize=allow",
        f"permit_id={permit['permit_id']}",
        f"reserved={permit['cost_reserved']}",
    )


def on_receipt(receipt: dict) -> None:
    print(
        "receipt=submitted",
        f"receipt_id={receipt['receipt_id']}",
        f"chain_hash={receipt['chain_hash']}",
    )


secure_charge = wrap_tool(
    charge_customer,
    AECPToolConfig(
        base_url=os.environ["AECP_BASE_URL"],
        delegation_token=os.environ["AECP_DELEGATION_TOKEN"],
        tool_id="tool.billing.charge",
        cost_estimate_fn=lambda args, kwargs: 1,
        action_fn=lambda args, kwargs: "charge_customer",
    ),
    callbacks=AECPCallbacks(on_authorize=on_authorize, on_receipt=on_receipt),
)


for i in range(2):
    payload = {"customer_id": "cus_123", "amount_cents": 1}
    try:
        result = secure_charge.invoke(payload)
        print(f"call_{i + 1}=executed result={result}")
    except AuthorizationDenied as exc:
        print(f"call_{i + 1}=denied fail_closed=true reason={exc}")
```

Expected behavior with budget limit `1`:

- Call 1: allowed, tool executes, receipt is emitted.
- Call 2: denied by AECP (`BUDGET_EXCEEDED`), tool does not execute (fail closed).

## Safety notes

AECP signs permits and receipts, and receipts are hash-chained per delegation. This gives auditors a tamper-evident execution trail while preserving a minimal integration surface for tool authors.

- AECP README: <AECP_REPO_README_URL>
- Signature and chain verification: <AECP_REPO_TRUST_MD_URL>
