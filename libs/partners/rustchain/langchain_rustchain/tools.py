"""RustChain Tool for LangChain.

RustChain is a DePIN Proof-of-Antiquity blockchain whose HTTP API is
agent-native (no auth, no captcha, wallet = any string). This integration
exposes it as a native LangChain ``BaseTool``.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlencode

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

_BASE_URL = os.environ.get("RUSTCHAIN_HOST", "https://rustchain.org")
_SSL_VERIFY = os.environ.get("RUSTCHAIN_SSL_VERIFY", "1").lower() in {"1", "true", "yes"}
_GITHUB_API = "https://api.github.com/repos/Scottcjn/rustchain-bounties/issues"
_BOUNTIES_LABEL = "bounty"


def _get(path: str, query: dict[str, str] | None = None) -> dict[str, Any] | list[Any]:
    url = f"{_BASE_URL}{path}"
    if query:
        url += f"?{urlencode(query)}"
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "langchain-rustchain/0.1.0"},
    )
    ctx = None
    if not _SSL_VERIFY:
        import ssl

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_bounties(limit: int) -> list[dict[str, Any]]:
    """Fetch bounty issues from the public GitHub REST API (no auth)."""
    url = f"{_GITHUB_API}?{urlencode({'labels': _BOUNTIES_LABEL, 'state': 'open', 'per_page': limit})}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "langchain-rustchain"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        issues = json.loads(resp.read().decode("utf-8"))
    return [
        {
            "number": i.get("number"),
            "title": i.get("title"),
            "url": i.get("html_url"),
            "labels": [l.get("name") for l in i.get("labels", [])],
        }
        for i in issues
        if isinstance(i, dict) and "pull_request" not in i
    ][:limit]


class RustChainTool(BaseTool):
    """RustChain (Proof-of-Antiquity DePIN) native LangChain tool.

    Setup:
        Install ``langchain-rustchain``.

        ```bash
        pip install -U langchain-rustchain
        ```

        The public node needs no API key. Optional environment variables:

        - ``RUSTCHAIN_HOST`` (default ``https://rustchain.org``)
        - ``RUSTCHAIN_SSL_VERIFY`` (``0`` to disable TLS verification for a
          self-signed node; see Limitations in the README)
        - ``GITHUB_TOKEN`` (raise the GitHub API rate limit for
          ``list_bounties`` from 60 to 5000 req/h)

    Instantiation:
        ```python
        from langchain_rustchain import RustChainTool

        tool = RustChainTool()
        ```

    Invocation:
        ``tool.invoke({"command": "get_node_health"})`` or directly call the
        public methods: ``check_balance``, ``list_bounties``,
        ``get_node_health``, ``get_current_epoch``.
    """

    name: str = "rustchain"
    description: str = (
        "Query the RustChain Proof-of-Antiquity blockchain: miner balance "
        "(check_balance), open bounties (list_bounties), node health "
        "(get_node_health), and the current epoch (get_current_epoch). "
        "Command format: JSON or NATLANG with a 'method' key; returns JSON."
    )

    def check_balance(self, wallet_id: str, run_manager: CallbackManagerForToolRun | None = None) -> float:
        """Return the RTC balance of a miner wallet (any string identifies it)."""
        data = _get("/wallet/balance", {"miner_id": wallet_id})
        if not isinstance(data, dict):
            return 0.0
        return float(data.get("amount_rtc", data.get("amount_i64", 0.0)))

    def list_bounties(self, limit: int = 10, run_manager: CallbackManagerForToolRun | None = None) -> list[dict[str, Any]]:
        """Return open bounties from the rustchain-bounties GitHub repo."""
        limit = max(1, min(int(limit), 100))
        return _get_bounties(limit)

    def get_node_health(self, run_manager: CallbackManagerForToolRun | None = None) -> dict[str, Any]:
        """Return the RustChain node health/version."""
        return _get("/health")

    def get_current_epoch(self, run_manager: CallbackManagerForToolRun | None = None) -> dict[str, Any]:
        """Return the current RustChain epoch and slot."""
        return _get("/epoch")

    def _run(self, tool_input: str | dict[str, Any], run_manager: CallbackManagerForToolRun | None = None) -> str:
        if isinstance(tool_input, str):
            try:
                tool_input = json.loads(tool_input)
            except json.JSONDecodeError:
                return json.dumps({"error": "invalid JSON input"})
        if not isinstance(tool_input, dict) or "method" not in tool_input:
            return json.dumps({"error": "missing 'method' key in tool input"})
        method = tool_input["method"]
        args = {k: v for k, v in tool_input.items() if k != "method"}
        if method == "check_balance":
            return json.dumps(self.check_balance(args.get("wallet_id", "demo")))
        if method == "list_bounties":
            return json.dumps(self.list_bounties(int(args.get("limit", 10))))
        if method == "get_node_health":
            return json.dumps(self.get_node_health())
        if method == "get_current_epoch":
            return json.dumps(self.get_current_epoch())
        return json.dumps({"error": f"unknown method: {method}"})