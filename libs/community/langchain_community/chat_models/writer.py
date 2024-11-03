"""Writer chat wrapper."""

from __future__ import annotations

import logging
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, ConfigDict, Field, SecretStr

logger = logging.getLogger(__name__)


def _convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a Writer message dict."""
    message_dict = {"role": "", "content": message.content}

    if isinstance(message, ChatMessage):
        message_dict["role"] = message.role
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
        if message.tool_calls:
            message_dict["tool_calls"] = [
                {
                    "id": tool["id"],
                    "type": "function",
                    "function": {"name": tool["name"], "arguments": tool["args"]},
                }
                for tool in message.tool_calls
            ]
    elif isinstance(message, SystemMessage):
        message_dict["role"] = "system"
    elif isinstance(message, ToolMessage):
        message_dict["role"] = "tool"
        message_dict["tool_call_id"] = message.tool_call_id
    else:
        raise ValueError(f"Got unknown message type: {type(message)}")

    if message.name:
        message_dict["name"] = message.name

    return message_dict


def _convert_dict_to_message(response_dict: Dict[str, Any]) -> BaseMessage:
    """Convert a Writer message dict to a LangChain message."""
    role = response_dict["role"]
    content = response_dict.get("content", "")

    if role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        additional_kwargs = {}
        if tool_calls := response_dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = tool_calls
        return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
    elif role == "system":
        return SystemMessage(content=content)
    elif role == "tool":
        return ToolMessage(
            content=content,
            tool_call_id=response_dict["tool_call_id"],
            name=response_dict.get("name"),
        )
    else:
        return ChatMessage(content=content, role=role)


class ChatWriter(BaseChatModel):
    """Writer chat model.

    To use, you should have the ``writer-sdk`` Python package installed, and the
    environment variable ``WRITER_API_KEY`` set with your API key.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatWriter

            chat = ChatWriter(model="palmyra-x-004")
    """

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    async_client: Any = Field(default=None, exclude=True)  #: :meta private:
    model_name: str = Field(default="palmyra-x-004", alias="model")
    """Model name to use."""
    temperature: float = 0.7
    """What sampling temperature to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    writer_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    """Writer API key."""
    writer_api_base: Optional[str] = Field(default=None, alias="base_url")
    """Base URL for API requests."""
    streaming: bool = False
    """Whether to stream the results or not."""
    n: int = 1
    """Number of chat completions to generate for each prompt."""
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""

    model_config = ConfigDict(populate_by_name=True)

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "writer-chat"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "streaming": self.streaming,
            **self.model_kwargs,
        }

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        generations = []
        for choice in response["choices"]:
            message = _convert_dict_to_message(choice["message"])
            gen = ChatGeneration(
                message=message,
                generation_info=dict(finish_reason=choice.get("finish_reason")),
            )
            generations.append(gen)

        token_usage = response.get("usage", {})
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model_name,
            "system_fingerprint": response.get("system_fingerprint", ""),
        }

        return ChatResult(generations=generations, llm_output=llm_output)

    def _convert_messages_to_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = {
            "model": self.model_name,
            "temperature": self.temperature,
            "n": self.n,
            "stream": self.streaming,
            **self.model_kwargs,
        }
        if stop:
            params["stop"] = stop
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens

        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._convert_messages_to_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        response = self.client.chat.chat(messages=message_dicts, **params)

        for chunk in response:
            delta = chunk["choices"][0].get("delta")
            if not delta or not delta.get("content"):
                continue
            chunk = _convert_dict_to_message(
                {"role": "assistant", "content": delta["content"]}
            )
            chunk = ChatGenerationChunk(message=chunk)

            if run_manager:
                run_manager.on_llm_new_token(chunk.text)

            yield chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._convert_messages_to_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        response = await self.async_client.chat.chat(messages=message_dicts, **params)

        async for chunk in response:
            delta = chunk["choices"][0].get("delta")
            if not delta or not delta.get("content"):
                continue
            chunk = _convert_dict_to_message(
                {"role": "assistant", "content": delta["content"]}
            )
            chunk = ChatGenerationChunk(message=chunk)

            if run_manager:
                await run_manager.on_llm_new_token(chunk.text)

            yield chunk

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            return generate_from_stream(
                self._stream(messages, stop, run_manager, **kwargs)
            )

        message_dicts, params = self._convert_messages_to_dicts(messages, stop)
        params = {**params, **kwargs}
        response = self.client.chat.chat(messages=message_dicts, **params)
        return self._create_chat_result(response)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            return await agenerate_from_stream(
                self._astream(messages, stop, run_manager, **kwargs)
            )

        message_dicts, params = self._convert_messages_to_dicts(messages, stop)
        params = {**params, **kwargs}
        response = await self.async_client.chat.chat(messages=message_dicts, **params)
        return self._create_chat_result(response)

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Writer API."""
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "stream": self.streaming,
            "n": self.n,
            "max_tokens": self.max_tokens,
            **self.model_kwargs,
        }

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable]],
        *,
        tool_choice: Optional[Union[str, Literal["auto", "none"]]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tools to the chat model.

        Args:
            tools: Tools to bind to the model
            tool_choice: Which tool to require ('auto', 'none', or specific tool name)
            **kwargs: Additional parameters to pass to the chat model

        Returns:
            A runnable that will use the tools
        """
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]

        if tool_choice:
            kwargs["tool_choice"] = (
                (tool_choice)
                if tool_choice in ("auto", "none")
                else {"type": "function", "function": {"name": tool_choice}}
            )

        return super().bind(tools=formatted_tools, **kwargs)
