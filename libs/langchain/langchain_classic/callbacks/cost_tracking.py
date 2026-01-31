from typing import Any, Dict, List, Optional
import time

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class CostTrackingCallback(BaseCallbackHandler):
    """Callback Handler that tracks cost and token usage."""

    def __init__(self, cost_per_1k_tokens: float = 0.002):
        self.cost_per_1k_tokens = cost_per_1k_tokens
        self.total_tokens = 0
        self.total_cost = 0.0
        self.start_time: Optional[float] = None

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        self.start_time = time.time()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        token_usage = response.llm_output.get("token_usage", {}) if response.llm_output else {}
        total_tokens = token_usage.get("total_tokens", 0)

        self.total_tokens += total_tokens
        cost = (total_tokens / 1000) * self.cost_per_1k_tokens
        self.total_cost += cost

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the cost and usage."""
        latency = 0.0
        if self.start_time is not None:
            latency = time.time() - self.start_time

        return {
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost, 6),
            "latency_sec": round(latency, 3),
        }
