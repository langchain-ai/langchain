"""Wrapper around Minimax chat models."""

import json
import logging
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Type, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env

logger = logging.getLogger(__name__)


@contextmanager
def connect_httpx_sse(client: Any, method: str, url: str, **kwargs: Any) -> Iterator:
    """Context manager for connecting to an SSE stream.

    Args:
        client: The httpx client.
        method: The HTTP method.
        url: The URL to connect to.
        kwargs: Additional keyword arguments to pass to the client.

    Yields:
        An EventSource object.
    """
    from httpx_sse import EventSource

    with client.stream(method, url, **kwargs) as response:
        yield EventSource(response)


@asynccontextmanager
async def aconnect_httpx_sse(
    client: Any, method: str, url: str, **kwargs: Any
) -> AsyncIterator:
    """Async context manager for connecting to an SSE stream.

    Args:
        client: The httpx client.
        method: The HTTP method.
        url: The URL to connect to.
        kwargs: Additional keyword arguments to pass to the client.

    Yields:
        An EventSource object.
    """
    from httpx_sse import EventSource

    async with client.stream(method, url, **kwargs) as response:
        yield EventSource(response)


def _convert_message_to_dict(message: BaseMessage) -> Dict[str, Any]:
    """Convert a LangChain messages to Dict."""
    message_dict: Dict[str, Any]
    if isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    else:
        raise TypeError(f"Got unknown type '{message.__class__.__name__}'.")
    return message_dict


def _convert_dict_to_message(dct: Dict[str, Any]) -> BaseMessage:
    """Convert a dict to LangChain message."""
    role = dct.get("role")
    content = dct.get("content", "")
    if role == "assistant":
        additional_kwargs = {}
        tool_calls = dct.get("tool_calls", None)
        if tool_calls is not None:
            additional_kwargs["tool_calls"] = tool_calls
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    return ChatMessage(role=role, content=content)  # type: ignore[arg-type]


def _convert_delta_to_message_chunk(
    dct: Dict[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = dct.get("role")
    content = dct.get("content", "")
    additional_kwargs = {}
    tool_calls = dct.get("tool_call", None)
    if tool_calls is not None:
        additional_kwargs["tool_calls"] = tool_calls

    if role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
    if role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)  # type: ignore[arg-type]
    return default_class(content=content)  # type: ignore[call-arg]


class MiniMaxChat(BaseChatModel):
    """MiniMax large language models.

    To use, you should have the environment variable``MINIMAX_API_KEY`` set with
    your API token, or pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import MiniMaxChat
            llm = MiniMaxChat(model="abab5-chat")

    """

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{"model": self.model}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "minimax"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            **self.model_kwargs,
        }

    _client: Any
    model: str = "abab6.5-chat"
    """Model name to use."""
    max_tokens: int = 256
    """Denotes the number of tokens to predict per generation."""
    temperature: float = 0.7
    """A non-negative float that tunes the degree of randomness in generation."""
    top_p: float = 0.95
    """Total probability mass of tokens to consider at each step."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    minimax_api_host: str = Field(
        default="https://api.minimax.chat/v1/text/chatcompletion_v2", alias="base_url"
    )
    minimax_group_id: Optional[str] = Field(default=None, alias="group_id")
    """[DEPRECATED, keeping it for for backward compatibility] Group Id"""
    minimax_api_key: SecretStr = Field(alias="api_key")
    """Minimax API Key"""
    streaming: bool = False
    """Whether to stream the results or not."""

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    @root_validator(pre=True, allow_reuse=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["minimax_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(
                values,
                ["minimax_api_key", "api_key"],
                "MINIMAX_API_KEY",
            )
        )
        values["minimax_group_id"] = get_from_dict_or_env(
            values, ["minimax_group_id", "group_id"], "MINIMAX_GROUP_ID"
        )
        # Get custom api url from environment.
        values["minimax_api_host"] = get_from_dict_or_env(
            values,
            "minimax_api_host",
            "MINIMAX_API_HOST",
            values["minimax_api_host"],
        )
        return values

    def _create_chat_result(self, response: Union[dict, BaseModel]) -> ChatResult:
        generations = []
        if not isinstance(response, dict):
            response = response.dict()
        for res in response["choices"]:
            message = _convert_dict_to_message(res["message"])
            generation_info = dict(finish_reason=res.get("finish_reason"))
            generations.append(
                ChatGeneration(message=message, generation_info=generation_info)
            )
        token_usage = response.get("usage", {})
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model,
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    def _create_payload_parameters(  # type: ignore[no-untyped-def]
        self, messages: List[BaseMessage], is_stream: bool = False, **kwargs
    ) -> Dict[str, Any]:
        """Create API request body parameters."""
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        payload = self._default_params
        payload["messages"] = message_dicts
        payload.update(**kwargs)
        if is_stream:
            payload["stream"] = True

        return payload

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate next turn in the conversation.
        Args:
            messages: The history of the conversation as a list of messages. Code chat
                does not support context.
            stop: The list of stop words (optional).
            run_manager: The CallbackManager for LLM run, it's not used at the moment.
            stream: Whether to stream the results or not.

        Returns:
            The ChatResult that contains outputs generated by the model.

        Raises:
            ValueError: if the last message in the list is not from human.
        """
        if not messages:
            raise ValueError(
                "You should provide at least one message to start the chat!"
            )
        is_stream = stream if stream is not None else self.streaming
        if is_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        payload = self._create_payload_parameters(messages, **kwargs)
        api_key = ""
        if self.minimax_api_key is not None:
            api_key = self.minimax_api_key.get_secret_value()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        import httpx

        with httpx.Client(headers=headers, timeout=60) as client:
            response = client.post(self.minimax_api_host, json=payload)
            response.raise_for_status()

        return self._create_chat_result(response.json())

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the chat response in chunks."""
        payload = self._create_payload_parameters(messages, is_stream=True, **kwargs)
        api_key = ""
        if self.minimax_api_key is not None:
            api_key = self.minimax_api_key.get_secret_value()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        import httpx

        with httpx.Client(headers=headers, timeout=60) as client:
            with connect_httpx_sse(
                client, "POST", self.minimax_api_host, json=payload
            ) as event_source:
                for sse in event_source.iter_sse():
                    chunk = json.loads(sse.data)
                    if len(chunk["choices"]) == 0:
                        continue
                    choice = chunk["choices"][0]
                    chunk = _convert_delta_to_message_chunk(
                        choice["delta"], AIMessageChunk
                    )
                    finish_reason = choice.get("finish_reason", None)

                    generation_info = (
                        {"finish_reason": finish_reason}
                        if finish_reason is not None
                        else None
                    )
                    chunk = ChatGenerationChunk(
                        message=chunk, generation_info=generation_info
                    )
                    yield chunk
                    if run_manager:
                        run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                    if finish_reason is not None:
                        break

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not messages:
            raise ValueError(
                "You should provide at least one message to start the chat!"
            )
        is_stream = stream if stream is not None else self.streaming
        if is_stream:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)
        payload = self._create_payload_parameters(messages, **kwargs)
        api_key = ""
        if self.minimax_api_key is not None:
            api_key = self.minimax_api_key.get_secret_value()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        import httpx

        async with httpx.AsyncClient(headers=headers, timeout=60) as client:
            response = await client.post(self.minimax_api_host, json=payload)
            response.raise_for_status()
        return self._create_chat_result(response.json())

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        payload = self._create_payload_parameters(messages, is_stream=True, **kwargs)
        api_key = ""
        if self.minimax_api_key is not None:
            api_key = self.minimax_api_key.get_secret_value()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        import httpx

        async with httpx.AsyncClient(headers=headers, timeout=60) as client:
            async with aconnect_httpx_sse(
                client, "POST", self.minimax_api_host, json=payload
            ) as event_source:
                async for sse in event_source.aiter_sse():
                    chunk = json.loads(sse.data)
                    if len(chunk["choices"]) == 0:
                        continue
                    choice = chunk["choices"][0]
                    chunk = _convert_delta_to_message_chunk(
                        choice["delta"], AIMessageChunk
                    )
                    finish_reason = choice.get("finish_reason", None)

                    generation_info = (
                        {"finish_reason": finish_reason}
                        if finish_reason is not None
                        else None
                    )
                    chunk = ChatGenerationChunk(
                        message=chunk, generation_info=generation_info
                    )
                    yield chunk
                    if run_manager:
                        await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                    if finish_reason is not None:
                        break
