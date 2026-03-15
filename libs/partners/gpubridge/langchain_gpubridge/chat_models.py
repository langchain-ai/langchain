"""GPU-Bridge Chat Model integration for LangChain."""

from typing import Any, Dict, Iterator, List, Optional

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, SecretStr, model_validator

GPUBRIDGE_API_URL = "https://api.gpubridge.xyz/run"


def _convert_messages_to_prompt(messages: List[BaseMessage]) -> str:
    """Convert LangChain messages to a single prompt string."""
    parts = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            parts.append(f"System: {msg.content}")
        elif isinstance(msg, HumanMessage):
            parts.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            parts.append(f"Assistant: {msg.content}")
        else:
            parts.append(str(msg.content))
    parts.append("Assistant:")
    return "\n".join(parts)


class ChatGPUBridge(BaseChatModel):
    """GPU-Bridge chat model.

    GPU-Bridge provides LLM inference (Llama, Mistral, Qwen, DeepSeek)
    with sub-second response times, powered by dedicated GPU infrastructure.

    Setup:
        Install with: pip install langchain-gpubridge

        .. code-block:: python

            from langchain_gpubridge import ChatGPUBridge

            llm = ChatGPUBridge(api_key="gpub_...", service="llm-4090")
            llm.invoke("Tell me about GPU inference")

    Key init args:
        api_key: GPU-Bridge API key. Register at https://gpubridge.xyz
        service: LLM service. Default ``llm-4090`` (Llama 3.3 70B).
        max_tokens: Max tokens to generate. Default 512.
        temperature: Sampling temperature. Default 0.7.
    """

    api_key: Optional[SecretStr] = Field(
        default=None,
        description="GPU-Bridge API key (starts with gpub_).",
    )
    service: str = Field(
        default="llm-4090",
        description="GPU-Bridge LLM service name.",
    )
    max_tokens: int = Field(default=512)
    temperature: float = Field(default=0.7)
    base_url: str = Field(default=GPUBRIDGE_API_URL)

    @property
    def _llm_type(self) -> str:
        return "gpu-bridge-chat"

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key.get_secret_value()}"
        return headers

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = _convert_messages_to_prompt(messages)

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

        text = data.get("output", {}).get("text", "")
        message = AIMessage(content=text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "service": self.service,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
