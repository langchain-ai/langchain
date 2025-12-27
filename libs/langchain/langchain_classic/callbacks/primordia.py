"""Primordia callback handler for economic metering."""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class PrimordiaCallbackHandler(BaseCallbackHandler):
    """Callback handler that emits MSR receipts for LLM usage.

    Operates in shadow mode by default (local only, no network).
    Set submit=True to index receipts for later netting.

    Example:
        >>> from langchain.callbacks import PrimordiaCallbackHandler
        >>> handler = PrimordiaCallbackHandler(agent_id="my-agent")
        >>> llm = ChatOpenAI(callbacks=[handler])
    """

    def __init__(
        self,
        agent_id: str,
        kernel_url: str = "https://clearing.kaledge.app",
        submit: bool = False,
    ):
        self.agent_id = agent_id
        self.kernel_url = kernel_url
        self.submit = submit
        self.receipts: List[Dict] = []

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Emit MSR receipt on LLM completion."""
        usage = response.llm_output.get("token_usage", {}) if response.llm_output else {}
        total_tokens = usage.get("total_tokens", 0)
        model = response.llm_output.get("model_name", "unknown") if response.llm_output else "unknown"

        # Estimate cost (adjust per model)
        unit_price = 50  # $0.00005 per token default
        if "gpt-4" in model:
            unit_price = 300
        elif "claude" in model:
            unit_price = 80

        receipt = {
            "meter_version": "0.1",
            "type": "compute",
            "agent_id": self.agent_id,
            "provider": model,
            "units": total_tokens,
            "unit_price_usd_micros": unit_price,
            "total_usd_micros": total_tokens * unit_price,
            "timestamp_ms": int(time.time() * 1000),
            "metadata": {"framework": "langchain", "run_id": str(run_id)}
        }

        receipt_hash = hashlib.sha256(
            json.dumps(receipt, sort_keys=True).encode()
        ).hexdigest()[:32]

        self.receipts.append({"hash": receipt_hash, "receipt": receipt})

        if self.submit:
            self._submit_receipt(receipt)

    def _submit_receipt(self, receipt: Dict) -> None:
        """Submit receipt to kernel for indexing (FREE)."""
        try:
            import requests
            requests.post(
                f"{self.kernel_url}/v1/index/batch",
                json={"agent_id": self.agent_id, "receipts": [receipt]},
                timeout=5
            )
        except Exception:
            pass  # Shadow mode - never block

    def get_receipts(self) -> List[Dict]:
        """Get all emitted receipts."""
        return self.receipts

    def get_total_usd(self) -> float:
        """Get total USD spent."""
        return sum(r["receipt"]["total_usd_micros"] for r in self.receipts) / 1_000_000
