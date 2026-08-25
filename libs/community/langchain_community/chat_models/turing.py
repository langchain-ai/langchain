"""
LangChain Chat Model Adapter for Turing Engine.
Enables native LangChain and LangGraph workflows with Turing Subspace Pruning controls.
"""

from typing import Any, Dict, List, Optional, Iterator
import json
import urllib.request

class ChatTuringEngine:
    """
    ChatTuringEngine provides native LangChain interface to local or remote Turing Engine.
    """
    def __init__(
        self,
        model: str = "llama-3.1-70b",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "turing-local",
        temperature: float = 0.7,
        max_tokens: int = 256,
        sparsity_ratio: float = 0.57,
        use_svd_kv: bool = True
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.sparsity_ratio = sparsity_ratio
        self.use_svd_kv = use_svd_kv

    def _format_messages(self, messages: Any) -> List[Dict[str, str]]:
        formatted = []
        for m in messages:
            if hasattr(m, "content"):
                role = getattr(m, "type", "user")
                if role == "human":
                    role = "user"
                elif role == "ai":
                    role = "assistant"
                formatted.append({"role": role, "content": m.content})
            elif isinstance(m, dict):
                formatted.append(m)
            elif isinstance(m, str):
                formatted.append({"role": "user", "content": m})
        return formatted

    def invoke(self, messages: Any, **kwargs) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": False
        }

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "X-Turing-Sparsity": str(self.sparsity_ratio)
            }
        )

        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        content = data["choices"][0]["message"]["content"]
        return {"content": content, "raw": data}
