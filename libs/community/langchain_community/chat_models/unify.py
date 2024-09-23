import json
import os
import posixpath
from operator import itemgetter
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
)

from httpx import (
    AsyncClient,
    AsyncHTTPTransport,
    Client,
    HTTPTransport,
    Limits,
    Response,
)
from langchain_core._api import beta
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
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
)
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import BaseModel, Field, SecretStr
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils import convert_to_secret_str, pre_init
from langchain_core.utils.function_calling import convert_to_openai_tool


class ChatUnify(BaseChatModel):
    """ChatUnify chat model.

    Example:
        .. code-block:: python

            from langchain_unify import ChatUnify


            model = ChatUnify(api_key="your-api-key")
            model.invoke("Hello, how are you?")

    """

    client: Client = Field(default=None)
    async_client: AsyncClient = Field(default=None)
    unify_api_key: Optional[SecretStr] = None
    unify_api_url: str = "https://api.unify.ai/v0/"
    max_retries: int = 5
    timeout: int = 120
    max_concurrent_requests: int = 128

    model: str = "llama-2-70b-chat@lowest-input-cost"

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "unify-chat"

    @property
    def _default_params(self) -> Dict[str, Any]:
        return {
            "model": self.model,
        }

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        unify_api_key = convert_to_secret_str(
            values.get("unify_api_key") or os.environ.get("UNIFY_KEY") or ""
        )
        values["unify_api_key"] = unify_api_key
        values["client"] = Client(
            timeout=values.get("timeout"),
            transport=HTTPTransport(retries=values.get("max_retries")),
        )
        values["async_client"] = AsyncClient(
            timeout=values.get("timeout"),
            limits=Limits(max_connections=values.get("max_concurrent_requests")),
            transport=AsyncHTTPTransport(retries=values.get("max_retries")),
        )
        return values

    def _check_response(self, response: Response) -> None:
        aread = False
        if isinstance(self.client, AsyncClient):
            aread = True
        if response.status_code >= 500:
            raise Exception(f"Unify Server: Error {response.status}")
        elif response.status_code >= 400:
            response.aread() if aread else response.read()
            raise ValueError(f"Unify received an invalid payload: {response.text}")
        elif response.status_code != 200:
            response.aread() if aread else response.read()
            raise Exception(
                f"Unify returned an unexpected response with status "
                f"{response.status}: {response.text}"
            )

    def _get_request_headers(self, stream) -> Dict[str, str]:
        return {
            "Accept": "text/event-stream" if stream else "application/json",
            "Authorization": f"Bearer {self.unify_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

    def _format_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        formatted_messages = []
        for message in messages:
            formatted_message = {}
            if isinstance(message, ChatMessage):
                formatted_message["role"] = message.role
            elif isinstance(message, HumanMessage):
                formatted_message["role"] = "user"
            elif isinstance(message, AIMessage):
                if "tool_calls" in message.additional_kwargs:
                    formatted_message["tool_calls"] = message.additional_kwargs[
                        "tool_calls"
                    ]
                formatted_message["role"] = "assistant"
            elif isinstance(message, SystemMessage):
                formatted_message["role"] = "system"
            elif isinstance(message, ToolMessage):
                formatted_message["role"] = "tool"
                formatted_message["name"] = message.name
            else:
                raise ValueError(f"Unsupported message type {message}")
            formatted_message["content"] = message.content
            formatted_messages.append(formatted_message)
        return formatted_messages

    def _convert_delta_to_message_chunk(
        self, _delta: Dict, default_class: Type[BaseMessageChunk]
    ) -> BaseMessageChunk:
        role = _delta.get("role")
        content = _delta.get("content", "")
        if role == "user" or default_class == HumanMessageChunk:
            return HumanMessageChunk(content=content)
        elif role == "assistant" or default_class == AIMessageChunk:
            additional_kwargs: Dict = {}
            if _delta.get("tool_calls"):
                additional_kwargs["tool_calls"] = _delta.get("tool_calls")
            return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
        elif role == "system" or default_class == SystemMessageChunk:
            return SystemMessageChunk(content=content)
        elif role or default_class == ChatMessageChunk:
            return ChatMessageChunk(content=content, role=role)
        else:
            return default_class(content=content)

    def parse_response_stream(self, default_chunk_class, line: str) -> ChatMessageChunk:
        response_dict = line.removeprefix("data: ")
        response_json = json.loads(response_dict)
        choices = response_json["choices"][0]
        if not choices:
            return None
        delta = choices["delta"]
        if not delta.get("content"):
            return None
        return self._convert_delta_to_message_chunk(delta, default_chunk_class)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        headers = self._get_request_headers(True)
        default_params = self._default_params
        body = {
            "messages": self._format_messages(messages),
            "stop": stop,
            "stream": True,
            **default_params,
            **kwargs,
        }
        url = posixpath.join(self.unify_api_url, "chat/completions")
        with self.client.stream("post", url, headers=headers, json=body) as response:
            self._check_response(response)
            default_chunk_class = AIMessageChunk
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = self.parse_response_stream(default_chunk_class, line)
                if not chunk:
                    continue
                default_chunk_class = chunk.__class__
                if run_manager:
                    run_manager.on_llm_new_token(token=chunk.content, chunk=chunk)
                yield ChatGenerationChunk(message=chunk)

    async def _acheck_response(self, response: Response) -> None:
        if response.status_code >= 500:
            raise Exception(f"Unify Server: Error {response.status}")
        elif response.status_code >= 400:
            raise ValueError(f"Unify received an invalid payload: {response.text}")
        elif response.status_code != 200:
            raise Exception(
                f"Unify returned an unexpected response with status "
                f"{response.status}: {response.text}"
            )

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        headers = self._get_request_headers(True)
        default_params = self._default_params
        body = {
            "messages": self._format_messages(messages),
            "stop": stop,
            "stream": True,
            **default_params,
            **kwargs,
        }
        url = posixpath.join(self.unify_api_url, "chat/completions")
        async with self.async_client.stream(
            "post", url, headers=headers, json=body
        ) as response:
            await self._acheck_response(response)
            default_chunk_class = AIMessageChunk
            async for line in response.aiter_lines():
                if not line:
                    continue
                chunk = self.parse_response_stream(default_chunk_class, line)
                if not chunk:
                    continue
                default_chunk_class = chunk.__class__

                if run_manager:
                    await run_manager.on_llm_new_token(token=chunk.content, chunk=chunk)
                yield ChatGenerationChunk(message=chunk)

    def _convert_to_message(self, _message: Dict[str, Any]) -> BaseMessage:
        role = _message.get("role")
        content = cast(Union[str, List], _message.get("content", ""))
        if role == "user":
            return HumanMessage(content=content)
        elif role == "assistant":
            additional_kwargs: Dict = {}
            if _message.get("tool_calls"):
                additional_kwargs["tool_calls"] = _message.get("tool_calls")
            return AIMessage(content=content, additional_kwargs=additional_kwargs)
        elif role == "system":
            return SystemMessage(content=content)
        elif role == "tool":
            return ToolMessage(content=content, name=_message.get("name"))
        else:
            return ChatMessage(content=content, role=role)

    def _format_output(self, data: Any, **kwargs: Any) -> ChatResult:
        generations = []
        for res in data["choices"]:
            finish_reason = res.get("finish_reason")
            gen = ChatGeneration(
                message=self._convert_to_message(res["message"]),
                generation_info={"finish_reason": finish_reason},
            )
            generations.append(gen)
        usage = data.get("usage")
        return ChatResult(
            generations=generations,
            llm_output={"usage": usage, "model": self.model},
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else False
        default_params = self._default_params
        params = {**default_params, **kwargs}
        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **params
            )
            return generate_from_stream(stream_iter)
        headers = self._get_request_headers(False)
        body = {"messages": self._format_messages(messages), **params}
        url = posixpath.join(self.unify_api_url, "chat/completions")
        response = self.client.post(url, headers=headers, json=body)
        self._check_response(response)
        return self._format_output(response.json(), **kwargs)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else False
        default_params = self._default_params
        params = {**default_params, **kwargs}
        if should_stream:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **params
            )
            return await agenerate_from_stream(stream_iter)
        headers = self._get_request_headers(False)
        body = {"messages": self._format_messages(messages), **params}
        url = posixpath.join(self.unify_api_url, "chat/completions")
        response = await self.async_client.post(url, headers=headers, json=body)
        self._check_response(response)
        return self._format_output(response.json(), **kwargs)

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return super().bind(tools=formatted_tools, **kwargs)

    @beta()
    def with_structured_output(
        self,
        schema: Union[Dict, Type[BaseModel]],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        is_pydantic_schema = isinstance(schema, type) and issubclass(schema, BaseModel)
        llm = self.bind_tools([schema], tool_choice="any")
        if is_pydantic_schema:
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
