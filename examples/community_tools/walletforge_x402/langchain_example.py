"""Safe WalletForge x402 LangChain example (unpaid 402 by default).

Paid tool invocation spends real Base USDC. This script only binds tools when
`BUYER_PRIVATE_KEY` is set, and it does not call the tools unless you pass
`--invoke`.

Usage:

    python examples/community_tools/walletforge_x402/langchain_example.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

API_BASE = "https://api.walletforge.app"


def unpaid_fetch_markdown_challenge() -> httpx.Response:
    """POST /v1/fetch-markdown without PAYMENT-SIGNATURE (HTTP 402, no spend)."""
    return httpx.post(
        f"{API_BASE}/v1/fetch-markdown",
        json={"url": "https://example.com", "max_chars": 5000},
        timeout=30.0,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--invoke",
        action="store_true",
        help="Invoke fetch_markdown (spends ~0.05 USDC on Base). Requires BUYER_PRIVATE_KEY.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    challenge = unpaid_fetch_markdown_challenge()
    print(f"unpaid fetch-markdown status={challenge.status_code}")
    body: dict[str, Any] = challenge.json()
    accepts = body.get("accepts") or []
    if accepts:
        print(
            "amount="
            f"{accepts[0].get('amount')} network={accepts[0].get('network')} "
            f"extensions={list((body.get('extensions') or {}).keys())}"
        )
    print(
        json.dumps(
            {"error": body.get("error"), "resource": body.get("resource")}, indent=2
        )
    )

    if challenge.status_code != 402:
        print("expected HTTP 402 for unpaid challenge", file=sys.stderr)
        return 1

    if not os.getenv("BUYER_PRIVATE_KEY"):
        print(
            "Set BUYER_PRIVATE_KEY to bind paid LangChain tools "
            "(will spend real Base USDC if invoked)."
        )
        return 0

    from examples.community_tools.walletforge_x402.tools import walletforge_tools

    tools = walletforge_tools()
    print("tools:", [getattr(tool, "name", type(tool).__name__) for tool in tools])
    if args.invoke:
        fetch_markdown = tools[0]
        print(fetch_markdown.invoke({"url": "https://example.com", "max_chars": 5000}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
