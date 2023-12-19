import os
from typing import Any, AsyncIterator, Dict, Iterator, List, Literal, Optional

import httpx
from httpx_sse import aconnect_sse, connect_sse
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator


class _AnthropicMessageContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


_message_type_lookups = {"human": "user", "assistant": "ai"}


class ChatAnthropicMessages(BaseChatModel):
    """Beta ChatAnthropicMessages chat model.

    Example:
        .. code-block:: python

            from langchain_anthropic import ChatAnthropicMessages

            model = ChatAnthropicMessages()
    """

    _client: httpx.Client
    _async_client: httpx.AsyncClient

    model: str = Field(alias="model_name")
    """Model name to use."""

    max_tokens: int = Field(default=256)
    """Denotes the number of tokens to predict per generation."""

    temperature: Optional[float] = None
    """A non-negative float that tunes the degree of randomness in generation."""

    top_k: Optional[int] = None
    """Number of most likely tokens to consider at each step."""

    top_p: Optional[float] = None
    """Total probability mass of tokens to consider at each step."""

    default_request_timeout: Optional[float] = None
    """Timeout for requests to Anthropic Completion API. Default is 600 seconds."""

    anthropic_api_url: str = "https://api.anthropic.com"

    anthropic_api_key: Optional[SecretStr] = None

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-anthropic-messages"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values["anthropic_api_key"] = values.get("anthropic_api_key") or os.environ.get(
            "ANTHROPIC_API_KEY"
        )
        values["_client"] = httpx.Client(
            base_url=values["anthropic_api_url"],
            timeout=values.get("default_request_timeout", 600),
            headers={
                "x-api-key": values["anthropic_api_key"],
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "messages-2023-12-15",
            },
        )
        values["_async_client"] = httpx.AsyncClient(
            base_url=values["anthropic_api_url"],
            timeout=values.get("default_request_timeout", 600),
            headers={
                "x-api-key": values["anthropic_api_key"],
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "messages-2023-12-15",
            },
        )
        return values

    def _request_body(
        self,
        *,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs: Dict,
    ) -> Dict:
        if any(not isinstance(m.content, str) for m in messages):
            raise ValueError("Anthropic Messages API only supports text generation.")
        if any(m.type not in _message_type_lookups for m in messages):
            raise ValueError(
                "Anthropic Messages API only supports user and ai messages."
            )
        rtn = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": _message_type_lookups[m.type],
                    "content": [_AnthropicMessageContent(text=m.content).dict()],
                }
                for m in messages
            ],
            "stop_sequences": stop,
            "stream": stream,
        }

        return rtn

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        with connect_sse(
            self._client,
            "POST",
            "/v1/messages",
            json=self._request_body(
                messages=messages, stop=stop, stream=True, **kwargs
            ),
        ) as sse:
            for event in sse.iter_sse():
                if event.event == "content_block_delta":
                    data = event.json()
                    content = data["delta"]["text"]
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(content=content),
                    )

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        async with aconnect_sse(
            self._async_client,
            "POST",
            "/v1/messages",
            json=self._request_body(
                messages=messages, stop=stop, stream=True, **kwargs
            ),
        ) as sse:
            async for event in sse.aiter_sse():
                if event.event == "content_block_delta":
                    data = event.json()
                    content = data["delta"]["text"]
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(content=content),
                    )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        res = self._client.post(
            "/v1/messages",
            json=self._request_body(messages=messages, stop=stop, **kwargs),
        )
        data = res.json()
        if res.status_code != 200:
            raise ValueError(str(data["error"]))

        content = data["content"][0]["text"]

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))],
            llm_output=data,
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        res = await self._async_client.post(
            "/v1/messages",
            json=self._request_body(messages=messages, stop=stop, **kwargs),
        )
        data = res.json()
        if res.status_code != 200:
            raise ValueError(str(data["error"]))
        content = data["content"][0]["text"]
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))],
            llm_output=data,
        )
