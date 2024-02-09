import os
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple

import anthropic
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
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str

_message_type_lookups = {"human": "user", "ai": "assistant"}


def _format_messages(messages: List[BaseMessage]) -> Tuple[Optional[str], List[Dict]]:
    """Format messages for anthropic."""

    """
    [
                {
                    "role": _message_type_lookups[m.type],
                    "content": [_AnthropicMessageContent(text=m.content).dict()],
                }
                for m in messages
            ]
    """
    system = None
    formatted_messages = []
    for i, message in enumerate(messages):
        if not isinstance(message.content, str):
            raise ValueError("Anthropic Messages API only supports text generation.")
        if message.type == "system":
            if i != 0:
                raise ValueError("System message must be at beginning of message list.")
            system = message.content
        else:
            formatted_messages.append(
                {
                    "role": _message_type_lookups[message.type],
                    "content": message.content,
                }
            )
    return system, formatted_messages


class ChatAnthropicMessages(BaseChatModel):
    """Beta ChatAnthropicMessages chat model.

    Example:
        .. code-block:: python

            from langchain_anthropic import ChatAnthropicMessages

            model = ChatAnthropicMessages()
    """

    _client: anthropic.Client = Field(default_factory=anthropic.Client)
    _async_client: anthropic.AsyncClient = Field(default_factory=anthropic.AsyncClient)

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

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-anthropic-messages"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        anthropic_api_key = convert_to_secret_str(
            values.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY") or ""
        )
        values["anthropic_api_key"] = anthropic_api_key
        values["_client"] = anthropic.Client(
            api_key=anthropic_api_key.get_secret_value()
        )
        values["_async_client"] = anthropic.AsyncClient(
            api_key=anthropic_api_key.get_secret_value()
        )
        return values

    def _format_params(
        self,
        *,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Dict,
    ) -> Dict:
        # get system prompt if any
        system, formatted_messages = _format_messages(messages)
        rtn = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "stop_sequences": stop,
            "system": system,
        }
        rtn = {k: v for k, v in rtn.items() if v is not None}

        return rtn

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        params = self._format_params(messages=messages, stop=stop, **kwargs)
        with self._client.beta.messages.stream(**params) as stream:
            for text in stream.text_stream:
                yield ChatGenerationChunk(message=AIMessageChunk(content=text))

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        params = self._format_params(messages=messages, stop=stop, **kwargs)
        async with self._async_client.beta.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                yield ChatGenerationChunk(message=AIMessageChunk(content=text))

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        params = self._format_params(messages=messages, stop=stop, **kwargs)
        data = self._client.beta.messages.create(**params)
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content=data.content[0].text))
            ],
            llm_output=data,
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        params = self._format_params(messages=messages, stop=stop, **kwargs)
        data = await self._async_client.beta.messages.create(**params)
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content=data.content[0].text))
            ],
            llm_output=data,
        )
