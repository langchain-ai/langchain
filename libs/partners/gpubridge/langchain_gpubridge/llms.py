"""GPU-Bridge LLM integration for LangChain."""

from typing import Any, Dict, Iterator, List, Optional

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.outputs import GenerationChunk
from pydantic import Field, SecretStr, model_validator

GPUBRIDGE_API_URL = "https://api.gpubridge.xyz/run"


class GPUBridgeLLM(LLM):
    """GPU-Bridge LLM inference.

    GPU-Bridge exposes 30 AI services — LLM, image, embeddings, STT, TTS,
    and more — through a single POST endpoint. Supports both API key auth
    and x402 autonomous payments (USDC on Base L2).

    Setup:
        Install with: pip install langchain-gpubridge

        .. code-block:: python

            from langchain_gpubridge import GPUBridgeLLM

            llm = GPUBridgeLLM(api_key="gpub_...", service="llm-4090")

    Key init args:
        api_key: GPU-Bridge API key. If not provided, x402 autonomous
            payment is attempted (requires a funded wallet).
        service: GPU-Bridge service name. Default is ``llm-4090``.
            See https://api.gpubridge.xyz/catalog for all options.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0.0 to 1.0).
        base_url: GPU-Bridge API base URL.
    """

    api_key: Optional[SecretStr] = Field(
        default=None,
        description="GPU-Bridge API key (starts with gpub_). "
        "Register at https://gpubridge.xyz",
    )
    service: str = Field(
        default="llm-4090",
        description="GPU-Bridge service name. See https://api.gpubridge.xyz/catalog",
    )
    max_tokens: int = Field(default=512, description="Maximum tokens to generate.")
    temperature: float = Field(default=0.7, description="Sampling temperature.")
    base_url: str = Field(
        default=GPUBRIDGE_API_URL,
        description="GPU-Bridge API endpoint.",
    )

    @model_validator(mode="after")
    def validate_api_key(self) -> "GPUBridgeLLM":
        """Warn if no API key is provided (x402 mode)."""
        if self.api_key is None:
            import warnings
            warnings.warn(
                "No GPU-Bridge API key provided. x402 autonomous payment mode "
                "requires a funded wallet at the PAYMENT_WALLET address. "
                "Register at https://gpubridge.xyz to get an API key.",
                UserWarning,
            )
        return self

    @property
    def _llm_type(self) -> str:
        return "gpu-bridge"

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key.get_secret_value()}"
        return headers

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        payload: Dict[str, Any] = {
            "service": self.service,
            "input": {
                "prompt": prompt,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
        }
        if stop:
            payload["input"]["stop"] = stop

        response = requests.post(
            self.base_url,
            json=payload,
            headers=self._get_headers(),
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            raise ValueError(f"GPU-Bridge error: {data['error']}")

        output = data.get("output", {})
        return output.get("text", str(output))

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "service": self.service,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "base_url": self.base_url,
        }
