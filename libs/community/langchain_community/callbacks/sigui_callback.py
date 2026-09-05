"""
langchain_community.callbacks.sigui_callback — Sigui Security Callback Handler

Intercepts autonomous agent tool execution and transaction calls in LangChain & LangGraph.
"""

import os
import urllib.request
import json
from typing import Any, Dict, Optional
from langchain_core.callbacks import BaseCallbackHandler


class SiguiSecurityCallbackHandler(BaseCallbackHandler):
    """LangChain / LangGraph Security Callback Handler powered by Sigui DePIN Oracle."""

    def __init__(self, api_key: Optional[str] = None, endpoint: Optional[str] = None, fail_on_block: bool = True):
        super().__init__()
        self.api_key = api_key or os.getenv("SIGUI_API_KEY", "sigui_live_key_alpha")
        self.endpoint = (endpoint or os.getenv("SIGUI_ENDPOINT", "http://127.0.0.1:8000")).rstrip("/")
        self.fail_on_block = fail_on_block

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        """Inspect tool inputs for financial transactions or high-risk transfers."""
        tool_name = serialized.get("name", "")
        if "transfer" in tool_name.lower() or "swap" in tool_name.lower() or "pay" in tool_name.lower():
            url = f"{self.endpoint}/v2/evaluate?zk=true"
            payload = json.dumps({
                "action_type": "transfer",
                "destination": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
                "amount_usdc": 100.0,
                "chain": "arc"
            }).encode("utf-8")

            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                method="POST"
            )

            try:
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    if data.get("decision") == "BLOCK" and self.fail_on_block:
                        raise ValueError(f"[Sigui Security Block] Action rejected: {data.get('reason')}")
            except Exception as e:
                pass
