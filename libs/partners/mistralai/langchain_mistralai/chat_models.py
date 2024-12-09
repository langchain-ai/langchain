from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from operator import itemgetter
from typing import (
    Any,
    AsyncContextManager,
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

import httpx
from httpx_sse import EventSource, aconnect_sse, connect_sse
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    LangSmithParams,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    InvalidToolCall,
    SystemMessage,
    SystemMessageChunk,
    ToolCall,
    ToolMessage,
)
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.output_parsers import (
    JsonOutputParser,
    PydanticOutputParser,
)
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils import secret_from_env
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.pydantic import is_basemodel_subclass
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    model_validator,
)
from typing_extensions import Self

logger = logging.getLogger(__name__)

# Mistral enforces a specific pattern for tool call IDs
TOOL_CALL_ID_PATTERN = re.compile(r"^[a-zA-Z0-9]{9}$")


def _create_retry_decorator(
    llm: ChatMistralAI,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:
    """Returns a tenacity retry decorator, preconfigured to handle exceptions"""

    errors = [httpx.RequestError, httpx.StreamError]
    return create_base_retry_decorator(
        error_types=errors, max_retries=llm.max_retries, run_manager=run_manager
    )


def _is_valid_mistral_tool_call_id(tool_call_id: str) -> bool:
    """Check if tool call ID is nine character string consisting of a-z, A-Z, 0-9"""
    return bool(TOOL_CALL_ID_PATTERN.match(tool_call_id))


def _base62_encode(num: int) -> str:
    """Encodes a number in base62 and ensures result is of a specified length."""
    base62 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if num == 0:
        return base62[0]
    arr = []
    base = len(base62)
    while num:
        num, rem = divmod(num, base)
        arr.append(base62[rem])
    arr.reverse()
    return "".join(arr)


def _convert_tool_call_id_to_mistral_compatible(tool_call_id: str) -> str:
    """Convert a tool call ID to a Mistral-compatible format"""
    if _is_valid_mistral_tool_call_id(tool_call_id):
        return tool_call_id
    else:
        hash_bytes = hashlib.sha256(tool_call_id.encode()).digest()
        hash_int = int.from_bytes(hash_bytes, byteorder="big")
        base62_str = _base62_encode(hash_int)
        if len(base62_str) >= 9:
            return base62_str[:9]
        else:
            return base62_str.rjust(9, "0")


def _convert_mistral_chat_message_to_message(
    _message: Dict,
) -> BaseMessage:
    role = _message["role"]
    assert role == "assistant", f"Expected role to be 'assistant', got {role}"
    content = cast(str, _message["content"])

    additional_kwargs: Dict = {}
    tool_calls = []
    invalid_tool_calls = []
    if raw_tool_calls := _message.get("tool_calls"):
        additional_kwargs["tool_calls"] = raw_tool_calls
        for raw_tool_call in raw_tool_calls:
            try:
                parsed: dict = cast(
                    dict, parse_tool_call(raw_tool_call, return_id=True)
                )
                if not parsed["id"]:
                    parsed["id"] = uuid.uuid4().hex[:]
                tool_calls.append(parsed)
            except Exception as e:
                invalid_tool_calls.append(make_invalid_tool_call(raw_tool_call, str(e)))
    return AIMessage(
        content=content,
        additional_kwargs=additional_kwargs,
        tool_calls=tool_calls,
        invalid_tool_calls=invalid_tool_calls,
    )


def _raise_on_error(response: httpx.Response) -> None:
    """Raise an error if the response is an error."""
    if httpx.codes.is_error(response.status_code):
        error_message = response.read().decode("utf-8")
        raise httpx.HTTPStatusError(
            f"Error response {response.status_code} "
            f"while fetching {response.url}: {error_message}",
            request=response.request,
            response=response,
        )


async def _araise_on_error(response: httpx.Response) -> None:
    """Raise an error if the response is an error."""
    if httpx.codes.is_error(response.status_code):
        error_message = (await response.aread()).decode("utf-8")
        raise httpx.HTTPStatusError(
            f"Error response {response.status_code} "
            f"while fetching {response.url}: {error_message}",
            request=response.request,
            response=response,
        )


async def _aiter_sse(
    event_source_mgr: AsyncContextManager[EventSource],
) -> AsyncIterator[Dict]:
    """Iterate over the server-sent events."""
    async with event_source_mgr as event_source:
        await _araise_on_error(event_source.response)
        async for event in event_source.aiter_sse():
            if event.data == "[DONE]":
                return
            yield event.json()


async def acompletion_with_retry(
    llm: ChatMistralAI,
    run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the async completion call."""
    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        if "stream" not in kwargs:
            kwargs["stream"] = False
        stream = kwargs["stream"]
        if stream:
            event_source = aconnect_sse(
                llm.async_client, "POST", "/chat/completions", json=kwargs
            )
            return _aiter_sse(event_source)
        else:
            response = await llm.async_client.post(url="/chat/completions", json=kwargs)
            await _araise_on_error(response)
            return response.json()

    return await _completion_with_retry(**kwargs)


def _convert_chunk_to_message_chunk(
    chunk: Dict, default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    _delta = chunk["choices"][0]["delta"]
    role = _delta.get("role")
    content = _delta.get("content") or ""
    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        additional_kwargs: Dict = {}
        if raw_tool_calls := _delta.get("tool_calls"):
            additional_kwargs["tool_calls"] = raw_tool_calls
            try:
                tool_call_chunks = []
                for raw_tool_call in raw_tool_calls:
                    if not raw_tool_call.get("index") and not raw_tool_call.get("id"):
                        tool_call_id = uuid.uuid4().hex[:]
                    else:
                        tool_call_id = raw_tool_call.get("id")
                    tool_call_chunks.append(
                        tool_call_chunk(
                            name=raw_tool_call["function"].get("name"),
                            args=raw_tool_call["function"].get("arguments"),
                            id=tool_call_id,
                            index=raw_tool_call.get("index"),
                        )
                    )
            except KeyError:
                pass
        else:
            tool_call_chunks = []
        if token_usage := chunk.get("usage"):
            usage_metadata = {
                "input_tokens": token_usage.get("prompt_tokens", 0),
                "output_tokens": token_usage.get("completion_tokens", 0),
                "total_tokens": token_usage.get("total_tokens", 0),
            }
        else:
            usage_metadata = None
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_call_chunks=tool_call_chunks,  # type: ignore[arg-type]
            usage_metadata=usage_metadata,  # type: ignore[arg-type]
        )
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)
    else:
        return default_class(content=content)  # type: ignore[call-arg]


def _format_tool_call_for_mistral(tool_call: ToolCall) -> dict:
    """Format Langchain ToolCall to dict expected by Mistral."""
    result: Dict[str, Any] = {
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["args"]),
        }
    }
    if _id := tool_call.get("id"):
        result["id"] = _convert_tool_call_id_to_mistral_compatible(_id)

    return result


def _format_invalid_tool_call_for_mistral(invalid_tool_call: InvalidToolCall) -> dict:
    """Format Langchain InvalidToolCall to dict expected by Mistral."""
    result: Dict[str, Any] = {
        "function": {
            "name": invalid_tool_call["name"],
            "arguments": invalid_tool_call["args"],
        }
    }
    if _id := invalid_tool_call.get("id"):
        result["id"] = _convert_tool_call_id_to_mistral_compatible(_id)

    return result


def _convert_message_to_mistral_chat_message(
    message: BaseMessage,
) -> Dict:
    if isinstance(message, ChatMessage):
        return dict(role=message.role, content=message.content)
    elif isinstance(message, HumanMessage):
        return dict(role="user", content=message.content)
    elif isinstance(message, AIMessage):
        message_dict: Dict[str, Any] = {"role": "assistant"}
        tool_calls = []
        if message.tool_calls or message.invalid_tool_calls:
            for tool_call in message.tool_calls:
                tool_calls.append(_format_tool_call_for_mistral(tool_call))
            for invalid_tool_call in message.invalid_tool_calls:
                tool_calls.append(
                    _format_invalid_tool_call_for_mistral(invalid_tool_call)
                )
        elif "tool_calls" in message.additional_kwargs:
            for tc in message.additional_kwargs["tool_calls"]:
                chunk = {
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    }
                }
                if _id := tc.get("id"):
                    chunk["id"] = _id
                tool_calls.append(chunk)
        else:
            pass
        if tool_calls:  # do not populate empty list tool_calls
            message_dict["tool_calls"] = tool_calls
        if tool_calls and message.content:
            # Assistant message must have either content or tool_calls, but not both.
            # Some providers may not support tool_calls in the same message as content.
            # This is done to ensure compatibility with messages from other providers.
            message_dict["content"] = ""
        else:
            message_dict["content"] = message.content
        return message_dict
    elif isinstance(message, SystemMessage):
        return dict(role="system", content=message.content)
    elif isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "content": message.content,
            "name": message.name,
            "tool_call_id": _convert_tool_call_id_to_mistral_compatible(
                message.tool_call_id
            ),
        }
    else:
        raise ValueError(f"Got unknown type {message}")


class ChatMistralAI(BaseChatModel):
    """A chat model that uses the MistralAI API."""

    # The type for client and async_client is ignored because the type is not
    # an Optional after the model is initialized and the model_validator
    # is run.
    client: httpx.Client = Field(  # type: ignore # : meta private:
        default=None, exclude=True
    )
    async_client: httpx.AsyncClient = Field(  # type: ignore # : meta private:
        default=None, exclude=True
    )  #: :meta private:
    mistral_api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env("MISTRAL_API_KEY", default=None),
    )
    endpoint: Optional[str] = Field(default=None, alias="base_url")
    max_retries: int = 5
    timeout: int = 120
    max_concurrent_requests: int = 64
    model: str = Field(default="mistral-small", alias="model_name")
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1
    """Decode using nucleus sampling: consider the smallest set of tokens whose
       probability sum is at least top_p. Must be in the closed interval [0.0, 1.0]."""
    random_seed: Optional[int] = None
    safe_mode: Optional[bool] = None
    streaming: bool = False

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling the API."""
        defaults = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "random_seed": self.random_seed,
            "safe_prompt": self.safe_mode,
        }
        filtered = {k: v for k, v in defaults.items() if v is not None}
        return filtered

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="mistral",
            ls_model_name=self.model,
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_max_tokens := params.get("max_tokens", self.max_tokens):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop", None):
            ls_params["ls_stop"] = ls_stop
        return ls_params

    @property
    def _client_params(self) -> Dict[str, Any]:
        """Get the parameters used for the client."""
        return self._default_params

    def completion_with_retry(
        self, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> Any:
        """Use tenacity to retry the completion call."""
        # retry_decorator = _create_retry_decorator(self, run_manager=run_manager)

        # @retry_decorator
        def _completion_with_retry(**kwargs: Any) -> Any:
            if "stream" not in kwargs:
                kwargs["stream"] = False
            stream = kwargs["stream"]
            if stream:

                def iter_sse() -> Iterator[Dict]:
                    with connect_sse(
                        self.client, "POST", "/chat/completions", json=kwargs
                    ) as event_source:
                        _raise_on_error(event_source.response)
                        for event in event_source.iter_sse():
                            if event.data == "[DONE]":
                                return
                            yield event.json()

                return iter_sse()
            else:
                response = self.client.post(url="/chat/completions", json=kwargs)
                _raise_on_error(response)
                return response.json()

        rtn = _completion_with_retry(**kwargs)
        return rtn

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        overall_token_usage: dict = {}
        for output in llm_outputs:
            if output is None:
                # Happens in streaming
                continue
            token_usage = output["token_usage"]
            if token_usage is not None:
                for k, v in token_usage.items():
                    if k in overall_token_usage:
                        overall_token_usage[k] += v
                    else:
                        overall_token_usage[k] = v
        combined = {"token_usage": overall_token_usage, "model_name": self.model}
        return combined

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate api key, python package exists, temperature, and top_p."""
        if isinstance(self.mistral_api_key, SecretStr):
            api_key_str: Optional[str] = self.mistral_api_key.get_secret_value()
        else:
            api_key_str = self.mistral_api_key

        # todo: handle retries
        base_url_str = (
            self.endpoint
            or os.environ.get("MISTRAL_BASE_URL")
            or "https://api.mistral.ai/v1"
        )
        self.endpoint = base_url_str
        if not self.client:
            self.client = httpx.Client(
                base_url=base_url_str,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {api_key_str}",
                },
                timeout=self.timeout,
            )
        # todo: handle retries and max_concurrency
        if not self.async_client:
            self.async_client = httpx.AsyncClient(
                base_url=base_url_str,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {api_key_str}",
                },
                timeout=self.timeout,
            )

        if self.temperature is not None and not 0 <= self.temperature <= 1:
            raise ValueError("temperature must be in the range [0.0, 1.0]")

        if self.top_p is not None and not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be in the range [0.0, 1.0]")

        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        response = self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        )
        return self._create_chat_result(response)

    def _create_chat_result(self, response: Dict) -> ChatResult:
        generations = []
        token_usage = response.get("usage", {})
        for res in response["choices"]:
            finish_reason = res.get("finish_reason")
            message = _convert_mistral_chat_message_to_message(res["message"])
            if token_usage and isinstance(message, AIMessage):
                message.usage_metadata = {
                    "input_tokens": token_usage.get("prompt_tokens", 0),
                    "output_tokens": token_usage.get("completion_tokens", 0),
                    "total_tokens": token_usage.get("total_tokens", 0),
                }
            gen = ChatGeneration(
                message=message,
                generation_info={"finish_reason": finish_reason},
            )
            generations.append(gen)

        llm_output = {"token_usage": token_usage, "model": self.model}
        return ChatResult(generations=generations, llm_output=llm_output)

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        params = self._client_params
        if stop is not None or "stop" in params:
            if "stop" in params:
                params.pop("stop")
            logger.warning(
                "Parameter `stop` not yet supported (https://docs.mistral.ai/api)"
            )
        message_dicts = [_convert_message_to_mistral_chat_message(m) for m in messages]
        return message_dicts, params

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class: Type[BaseMessageChunk] = AIMessageChunk
        for chunk in self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        ):
            if len(chunk["choices"]) == 0:
                continue
            new_chunk = _convert_chunk_to_message_chunk(chunk, default_chunk_class)
            # make future chunks same type as first chunk
            default_chunk_class = new_chunk.__class__
            gen_chunk = ChatGenerationChunk(message=new_chunk)
            if run_manager:
                run_manager.on_llm_new_token(
                    token=cast(str, new_chunk.content), chunk=gen_chunk
                )
            yield gen_chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class: Type[BaseMessageChunk] = AIMessageChunk
        async for chunk in await acompletion_with_retry(
            self, messages=message_dicts, run_manager=run_manager, **params
        ):
            if len(chunk["choices"]) == 0:
                continue
            new_chunk = _convert_chunk_to_message_chunk(chunk, default_chunk_class)
            # make future chunks same type as first chunk
            default_chunk_class = new_chunk.__class__
            gen_chunk = ChatGenerationChunk(message=new_chunk)
            if run_manager:
                await run_manager.on_llm_new_token(
                    token=cast(str, new_chunk.content), chunk=gen_chunk
                )
            yield gen_chunk

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._astream(
                messages=messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        response = await acompletion_with_retry(
            self, messages=message_dicts, run_manager=run_manager, **params
        )
        return self._create_chat_result(response)

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with OpenAI tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Supports any tool definition handled by
                :meth:`langchain_core.utils.function_calling.convert_to_openai_tool`.
            tool_choice: Which tool to require the model to call.
                Must be the name of the single provided function or
                "auto" to automatically determine which function to call
                (if any), or a dict of the form:
                {"type": "function", "function": {"name": <<tool_name>>}}.
            kwargs: Any additional parameters are passed directly to
                ``self.bind(**kwargs)``.
        """

        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: Optional[Union[Dict, Type]] = None,
        *,
        method: Literal["function_calling", "json_mode"] = "function_calling",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema:
                The output schema. Can be passed in as:
                    - an OpenAI function/tool schema,
                    - a JSON Schema,
                    - a TypedDict class (support added in 0.1.12),
                    - or a Pydantic class.
                If ``schema`` is a Pydantic class then the model output will be a
                Pydantic instance of that class, and the model-generated fields will be
                validated by the Pydantic class. Otherwise the model output will be a
                dict and will not be validated. See :meth:`langchain_core.utils.function_calling.convert_to_openai_tool`
                for more on how to properly specify types and descriptions of
                schema fields when specifying a Pydantic or TypedDict class.

                .. versionchanged:: 0.1.12

                        Added support for TypedDict class.

            method:
                The method for steering model generation, either "function_calling"
                or "json_mode". If "function_calling" then the schema will be converted
                to an OpenAI function and the returned model will make use of the
                function-calling API. If "json_mode" then OpenAI's JSON mode will be
                used. Note that if using "json_mode" then you must include instructions
                for formatting the output into the desired schema into the model call.
            include_raw:
                If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys "raw", "parsed", and "parsing_error".

        Returns:
            A Runnable that takes same inputs as a :class:`langchain_core.language_models.chat.BaseChatModel`.

            If ``include_raw`` is False and ``schema`` is a Pydantic class, Runnable outputs
            an instance of ``schema`` (i.e., a Pydantic object).

            Otherwise, if ``include_raw`` is False then Runnable outputs a dict.

            If ``include_raw`` is True, then Runnable outputs a dict with keys:
                - ``"raw"``: BaseMessage
                - ``"parsed"``: None if there was a parsing error, otherwise the type depends on the ``schema`` as described above.
                - ``"parsing_error"``: Optional[BaseException]

        Example: schema=Pydantic class, method="function_calling", include_raw=False:
            .. code-block:: python

                from typing import Optional

                from langchain_mistralai import ChatMistralAI
                from pydantic import BaseModel, Field


                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    # If we provide default values and/or descriptions for fields, these will be passed
                    # to the model. This is an important part of improving a model's ability to
                    # correctly return structured outputs.
                    justification: Optional[str] = Field(
                        default=None, description="A justification for the answer."
                    )


                llm = ChatMistralAI(model="mistral-large-latest", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification)

                structured_llm.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )

                # -> AnswerWithJustification(
                #     answer='They weigh the same',
                #     justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
                # )

        Example: schema=Pydantic class, method="function_calling", include_raw=True:
            .. code-block:: python

                from langchain_mistralai import ChatMistralAI
                from pydantic import BaseModel


                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    justification: str


                llm = ChatMistralAI(model="mistral-large-latest", temperature=0)
                structured_llm = llm.with_structured_output(
                    AnswerWithJustification, include_raw=True
                )

                structured_llm.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )
                # -> {
                #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Ao02pnFYXD6GN1yzc0uXPsvF', 'function': {'arguments': '{"answer":"They weigh the same.","justification":"Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ."}', 'name': 'AnswerWithJustification'}, 'type': 'function'}]}),
                #     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'),
                #     'parsing_error': None
                # }

        Example: schema=TypedDict class, method="function_calling", include_raw=False:
            .. code-block:: python

                # IMPORTANT: If you are using Python <=3.8, you need to import Annotated
                # from typing_extensions, not from typing.
                from typing_extensions import Annotated, TypedDict

                from langchain_mistralai import ChatMistralAI


                class AnswerWithJustification(TypedDict):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    justification: Annotated[
                        Optional[str], None, "A justification for the answer."
                    ]


                llm = ChatMistralAI(model="mistral-large-latest", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification)

                structured_llm.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
                # }

        Example: schema=OpenAI function schema, method="function_calling", include_raw=False:
            .. code-block:: python

                from langchain_mistralai import ChatMistralAI

                oai_schema = {
                    'name': 'AnswerWithJustification',
                    'description': 'An answer to the user question along with justification for the answer.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'answer': {'type': 'string'},
                            'justification': {'description': 'A justification for the answer.', 'type': 'string'}
                        },
                       'required': ['answer']
                   }
               }

                llm = ChatMistralAI(model="mistral-large-latest", temperature=0)
                structured_llm = llm.with_structured_output(oai_schema)

                structured_llm.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
                # }

        Example: schema=Pydantic class, method="json_mode", include_raw=True:
            .. code-block::

                from langchain_mistralai import ChatMistralAI
                from pydantic import BaseModel

                class AnswerWithJustification(BaseModel):
                    answer: str
                    justification: str

                llm = ChatMistralAI(model="mistral-large-latest", temperature=0)
                structured_llm = llm.with_structured_output(
                    AnswerWithJustification,
                    method="json_mode",
                    include_raw=True
                )

                structured_llm.invoke(
                    "Answer the following question. "
                    "Make sure to return a JSON blob with keys 'answer' and 'justification'.\n\n"
                    "What's heavier a pound of bricks or a pound of feathers?"
                )
                # -> {
                #     'raw': AIMessage(content='{\n    "answer": "They are both the same weight.",\n    "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight." \n}'),
                #     'parsed': AnswerWithJustification(answer='They are both the same weight.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight.'),
                #     'parsing_error': None
                # }

        Example: schema=None, method="json_mode", include_raw=True:
            .. code-block::

                structured_llm = llm.with_structured_output(method="json_mode", include_raw=True)

                structured_llm.invoke(
                    "Answer the following question. "
                    "Make sure to return a JSON blob with keys 'answer' and 'justification'.\n\n"
                    "What's heavier a pound of bricks or a pound of feathers?"
                )
                # -> {
                #     'raw': AIMessage(content='{\n    "answer": "They are both the same weight.",\n    "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight." \n}'),
                #     'parsed': {
                #         'answer': 'They are both the same weight.',
                #         'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight.'
                #     },
                #     'parsing_error': None
                # }
        """  # noqa: E501
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        is_pydantic_schema = isinstance(schema, type) and is_basemodel_subclass(schema)
        if method == "function_calling":
            if schema is None:
                raise ValueError(
                    "schema must be specified when method is 'function_calling'. "
                    "Received None."
                )
            # TODO: Update to pass in tool name as tool_choice if/when Mistral supports
            # specifying a tool.
            llm = self.bind_tools([schema], tool_choice="any")
            if is_pydantic_schema:
                output_parser: OutputParserLike = PydanticToolsParser(
                    tools=[schema],  # type: ignore[list-item]
                    first_tool_only=True,  # type: ignore[list-item]
                )
            else:
                key_name = convert_to_openai_tool(schema)["function"]["name"]
                output_parser = JsonOutputKeyToolsParser(
                    key_name=key_name, first_tool_only=True
                )
        elif method == "json_mode":
            llm = self.bind(response_format={"type": "json_object"})
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[type-var, arg-type]
                if is_pydantic_schema
                else JsonOutputParser()
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

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return self._default_params

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "mistralai-chat"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"mistral_api_key": "MISTRAL_API_KEY"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "mistralai"]
