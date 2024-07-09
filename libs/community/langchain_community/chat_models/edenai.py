import json
import warnings
from operator import itemgetter
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)

from aiohttp import ClientSession
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
    HumanMessage,
    InvalidToolCall,
    SystemMessage,
    ToolCall,
    ToolCallChunk,
    ToolMessage,
)
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import (
    BaseModel,
    Extra,
    Field,
    SecretStr,
)
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env, pre_init
from langchain_core.utils.function_calling import convert_to_openai_tool

from langchain_community.utilities.requests import Requests


def _result_to_chunked_message(generated_result: ChatResult) -> ChatGenerationChunk:
    message = generated_result.generations[0].message
    if isinstance(message, AIMessage) and message.tool_calls is not None:
        tool_call_chunks = [
            ToolCallChunk(
                name=tool_call["name"],
                args=json.dumps(tool_call["args"]),
                id=tool_call["id"],
                index=idx,
            )
            for idx, tool_call in enumerate(message.tool_calls)
        ]
        message_chunk = AIMessageChunk(
            content=message.content,
            tool_call_chunks=tool_call_chunks,
        )
        return ChatGenerationChunk(message=message_chunk)
    else:
        return cast(ChatGenerationChunk, generated_result.generations[0])


def _message_role(type: str) -> str:
    role_mapping = {
        "ai": "assistant",
        "human": "user",
        "chat": "user",
        "AIMessageChunk": "assistant",
    }

    if type in role_mapping:
        return role_mapping[type]
    else:
        raise ValueError(f"Unknown type: {type}")


def _extract_edenai_tool_results_from_messages(
    messages: List[BaseMessage],
) -> Tuple[List[Dict[str, Any]], List[BaseMessage]]:
    """
    Get the last langchain tools messages to transform them into edenai tool_results
    Returns tool_results and messages without the extracted tool messages
    """
    tool_results: List[Dict[str, Any]] = []
    other_messages = messages[:]
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            tool_results = [
                {"id": msg.tool_call_id, "result": msg.content},
                *tool_results,
            ]
            other_messages.pop()
        else:
            break
    return tool_results, other_messages


def _format_edenai_messages(messages: List[BaseMessage]) -> Dict[str, Any]:
    system = None
    formatted_messages = []

    human_messages = filter(lambda msg: isinstance(msg, HumanMessage), messages)
    last_human_message = list(human_messages)[-1] if human_messages else ""

    tool_results, other_messages = _extract_edenai_tool_results_from_messages(messages)
    for i, message in enumerate(other_messages):
        if isinstance(message, SystemMessage):
            if i != 0:
                raise ValueError("System message must be at beginning of message list.")
            system = message.content
        elif isinstance(message, ToolMessage):
            formatted_messages.append({"role": "tool", "message": message.content})
        elif message != last_human_message:
            formatted_messages.append(
                {
                    "role": _message_role(message.type),
                    "message": message.content,
                    "tool_calls": _format_tool_calls_to_edenai_tool_calls(message),
                }
            )

    return {
        "text": getattr(last_human_message, "content", ""),
        "previous_history": formatted_messages,
        "chatbot_global_action": system,
        "tool_results": tool_results,
    }


def _format_tool_calls_to_edenai_tool_calls(message: BaseMessage) -> List:
    tool_calls = getattr(message, "tool_calls", [])
    invalid_tool_calls = getattr(message, "invalid_tool_calls", [])
    edenai_tool_calls = []

    for invalid_tool_call in invalid_tool_calls:
        edenai_tool_calls.append(
            {
                "arguments": invalid_tool_call.get("args"),
                "id": invalid_tool_call.get("id"),
                "name": invalid_tool_call.get("name"),
            }
        )

    for tool_call in tool_calls:
        tool_args = tool_call.get("args", {})
        try:
            arguments = json.dumps(tool_args)
        except TypeError:
            arguments = str(tool_args)
        edenai_tool_calls.append(
            {
                "arguments": arguments,
                "id": tool_call["id"],
                "name": tool_call["name"],
            }
        )
    return edenai_tool_calls


def _extract_tool_calls_from_edenai_response(
    provider_response: Dict[str, Any],
) -> Tuple[List[ToolCall], List[InvalidToolCall]]:
    tool_calls = []
    invalid_tool_calls = []

    message = provider_response.get("message", {})[1]

    if raw_tool_calls := message.get("tool_calls"):
        for raw_tool_call in raw_tool_calls:
            try:
                tool_calls.append(
                    ToolCall(
                        name=raw_tool_call["name"],
                        args=json.loads(raw_tool_call["arguments"]),
                        id=raw_tool_call["id"],
                    )
                )
            except json.JSONDecodeError as exc:
                invalid_tool_calls.append(
                    InvalidToolCall(
                        name=raw_tool_call.get("name"),
                        args=raw_tool_call.get("arguments"),
                        id=raw_tool_call.get("id"),
                        error=f"Received JSONDecodeError {exc}",
                    )
                )

    return tool_calls, invalid_tool_calls


class ChatEdenAI(BaseChatModel):
    """`EdenAI` chat large language models.

    `EdenAI` is a versatile platform that allows you to access various language models
    from different providers such as Google, OpenAI, Cohere, Mistral and more.

    To get started, make sure you have the environment variable ``EDENAI_API_KEY``
    set with your API key, or pass it as a named parameter to the constructor.

    Additionally, `EdenAI` provides the flexibility to choose from a variety of models,
    including the ones like "gpt-4".

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatEdenAI
            from langchain_core.messages import HumanMessage

            # Initialize `ChatEdenAI` with the desired configuration
            chat = ChatEdenAI(
                provider="openai",
                model="gpt-4",
                max_tokens=256,
                temperature=0.75)

            # Create a list of messages to interact with the model
            messages = [HumanMessage(content="hello")]

            # Invoke the model with the provided messages
            chat.invoke(messages)

    `EdenAI` goes beyond mere model invocation. It empowers you with advanced features :

    - **Multiple Providers**: access to a diverse range of llms offered by various
     providers giving you the freedom to choose the best-suited model for your use case.

    - **Fallback Mechanism**: Set a fallback mechanism to ensure seamless operations
        even if the primary provider is unavailable, you can easily switches to an
        alternative provider.

    - **Usage Statistics**: Track usage statistics on a per-project
    and per-API key basis.
    This feature allows you to monitor and manage resource consumption effectively.

    - **Monitoring and Observability**: `EdenAI` provides comprehensive monitoring
    and observability tools on the platform.

    Example of setting up a fallback mechanism:
        .. code-block:: python

            # Initialize `ChatEdenAI` with a fallback provider
            chat_with_fallback = ChatEdenAI(
                provider="openai",
                model="gpt-4",
                max_tokens=256,
                temperature=0.75,
                fallback_provider="google")

    you can find more details here : https://docs.edenai.co/reference/text_chat_create
    """

    provider: str = "openai"
    """chat provider to use (eg: openai,google etc.)"""

    model: Optional[str] = None
    """
    model name for above provider (eg: 'gpt-4' for openai)
    available models are shown on https://docs.edenai.co/ under 'available providers'
    """

    max_tokens: int = 256
    """Denotes the number of tokens to predict per generation."""

    temperature: Optional[float] = 0
    """A non-negative float that tunes the degree of randomness in generation."""

    streaming: bool = False
    """Whether to stream the results."""

    fallback_providers: Optional[str] = None
    """Providers in this will be used as fallback if the call to provider fails."""

    edenai_api_url: str = "https://api.edenai.run/v2"

    edenai_api_key: Optional[SecretStr] = Field(None, description="EdenAI API Token")

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        values["edenai_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "edenai_api_key", "EDENAI_API_KEY")
        )
        return values

    @staticmethod
    def get_user_agent() -> str:
        from langchain_community import __version__

        return f"langchain/{__version__}"

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "edenai-chat"

    @property
    def _api_key(self) -> str:
        if self.edenai_api_key:
            return self.edenai_api_key.get_secret_value()
        return ""

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Call out to EdenAI's chat endpoint."""
        if "available_tools" in kwargs:
            yield self._stream_with_tools_as_generate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return
        url = f"{self.edenai_api_url}/text/chat/stream"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": self.get_user_agent(),
        }
        formatted_data = _format_edenai_messages(messages=messages)
        payload: Dict[str, Any] = {
            "providers": self.provider,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "fallback_providers": self.fallback_providers,
            **formatted_data,
            **kwargs,
        }

        payload = {k: v for k, v in payload.items() if v is not None}

        if self.model is not None:
            payload["settings"] = {self.provider: self.model}

        request = Requests(headers=headers)
        response = request.post(url=url, data=payload, stream=True)
        response.raise_for_status()

        for chunk_response in response.iter_lines():
            chunk = json.loads(chunk_response.decode())
            token = chunk["text"]
            cg_chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=cg_chunk)
            yield cg_chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if "available_tools" in kwargs:
            yield await self._astream_with_tools_as_agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return
        url = f"{self.edenai_api_url}/text/chat/stream"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": self.get_user_agent(),
        }
        formatted_data = _format_edenai_messages(messages=messages)
        payload: Dict[str, Any] = {
            "providers": self.provider,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "fallback_providers": self.fallback_providers,
            **formatted_data,
            **kwargs,
        }

        payload = {k: v for k, v in payload.items() if v is not None}

        if self.model is not None:
            payload["settings"] = {self.provider: self.model}

        async with ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                response.raise_for_status()
                async for chunk_response in response.content:
                    chunk = json.loads(chunk_response.decode())
                    token = chunk["text"]
                    cg_chunk = ChatGenerationChunk(
                        message=AIMessageChunk(content=token)
                    )
                    if run_manager:
                        await run_manager.on_llm_new_token(
                            token=chunk["text"], chunk=cg_chunk
                        )
                    yield cg_chunk

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "none", "required", "any"], bool]
        ] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        formatted_tools = [convert_to_openai_tool(tool)["function"] for tool in tools]
        formatted_tool_choice = "required" if tool_choice == "any" else tool_choice
        return super().bind(
            available_tools=formatted_tools, tool_choice=formatted_tool_choice, **kwargs
        )

    def with_structured_output(
        self,
        schema: Union[Dict, Type[BaseModel]],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        llm = self.bind_tools([schema], tool_choice="required")
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            output_parser: OutputParserLike = PydanticToolsParser(
                tools=[schema], first_tool_only=True
            )
        else:
            key_name = convert_to_openai_tool(schema)["function"]["name"]
            output_parser = JsonOutputKeyToolsParser(
                key_name=key_name, first_tool_only=True
            )

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        else:
            return llm | output_parser

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call out to EdenAI's chat endpoint."""
        if self.streaming:
            if "available_tools" in kwargs:
                warnings.warn(
                    "stream: Tool use is not yet supported in streaming mode."
                )
            else:
                stream_iter = self._stream(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
                return generate_from_stream(stream_iter)

        url = f"{self.edenai_api_url}/text/chat"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": self.get_user_agent(),
        }
        formatted_data = _format_edenai_messages(messages=messages)

        payload: Dict[str, Any] = {
            "providers": self.provider,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "fallback_providers": self.fallback_providers,
            **formatted_data,
            **kwargs,
        }

        payload = {k: v for k, v in payload.items() if v is not None}

        if self.model is not None:
            payload["settings"] = {self.provider: self.model}

        request = Requests(headers=headers)
        response = request.post(url=url, data=payload)

        response.raise_for_status()
        data = response.json()
        provider_response = data[self.provider]

        if self.fallback_providers:
            fallback_response = data.get(self.fallback_providers)
            if fallback_response:
                provider_response = fallback_response

        if provider_response.get("status") == "fail":
            err_msg = provider_response.get("error", {}).get("message")
            raise Exception(err_msg)

        tool_calls, invalid_tool_calls = _extract_tool_calls_from_edenai_response(
            provider_response
        )

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=provider_response["generated_text"] or "",
                        tool_calls=tool_calls,
                        invalid_tool_calls=invalid_tool_calls,
                    )
                )
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
        if self.streaming:
            if "available_tools" in kwargs:
                warnings.warn(
                    "stream: Tool use is not yet supported in streaming mode."
                )
            else:
                stream_iter = self._astream(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
                return await agenerate_from_stream(stream_iter)

        url = f"{self.edenai_api_url}/text/chat"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": self.get_user_agent(),
        }
        formatted_data = _format_edenai_messages(messages=messages)
        payload: Dict[str, Any] = {
            "providers": self.provider,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "fallback_providers": self.fallback_providers,
            **formatted_data,
            **kwargs,
        }

        payload = {k: v for k, v in payload.items() if v is not None}

        if self.model is not None:
            payload["settings"] = {self.provider: self.model}

        async with ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
                provider_response = data[self.provider]

                if self.fallback_providers:
                    fallback_response = data.get(self.fallback_providers)
                    if fallback_response:
                        provider_response = fallback_response

                if provider_response.get("status") == "fail":
                    err_msg = provider_response.get("error", {}).get("message")
                    raise Exception(err_msg)

                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content=provider_response["generated_text"]
                            )
                        )
                    ],
                    llm_output=data,
                )

    def _stream_with_tools_as_generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]],
        run_manager: Optional[CallbackManagerForLLMRun],
        **kwargs: Any,
    ) -> ChatGenerationChunk:
        warnings.warn("stream: Tool use is not yet supported in streaming mode.")
        result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        return _result_to_chunked_message(result)

    async def _astream_with_tools_as_agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]],
        run_manager: Optional[AsyncCallbackManagerForLLMRun],
        **kwargs: Any,
    ) -> ChatGenerationChunk:
        warnings.warn("stream: Tool use is not yet supported in streaming mode.")
        result = await self._agenerate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
        return _result_to_chunked_message(result)
