"""Baseten chat wrapper."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator, Mapping
from operator import itemgetter
from typing import Any, Callable, Literal, Optional, Union

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
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    InvalidToolCall,
    SystemMessage,
    SystemMessageChunk,
    ToolCall,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils import (
    get_pydantic_field_names,
)
from langchain_core.utils.function_calling import (
    convert_to_json_schema,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import is_basemodel_subclass
from langchain_core.utils.utils import _build_model_kwargs, secret_from_env
from openai import AsyncOpenAI, OpenAI
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    model_validator,
)
from typing_extensions import Self

logger = logging.getLogger(__name__)


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """Convert a dictionary to a LangChain message.

    Args:
        _dict: The dictionary.

    Returns:
        The LangChain message.
    """
    role = _dict.get("role")
    if role == "user":
        return HumanMessage(content=_dict.get("content", ""))
    if role == "assistant":
        content = _dict.get("content") or ""
        additional_kwargs: dict[str, Any] = {}
        if tool_calls := _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = tool_calls
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    if role == "system":
        return SystemMessage(content=_dict.get("content", ""))
    if role == "function":
        return FunctionMessage(
            content=_dict.get("content", ""), name=_dict.get("name", "")
        )
    if role == "tool":
        return ToolMessage(
            content=_dict.get("content", ""),
            tool_call_id=_dict.get("tool_call_id", ""),
        )
    return ChatMessage(content=_dict.get("content", ""), role=role)


def _convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """
    message_dict: dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
        if message.tool_calls or message.invalid_tool_calls:
            message_dict["tool_calls"] = [
                _lc_tool_call_to_openai_tool_call(tc) for tc in message.tool_calls
            ] + [
                _lc_invalid_tool_call_to_openai_tool_call(tc)
                for tc in message.invalid_tool_calls
            ]
        elif "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
        # If tool calls present, content null value should be None not empty string.
        if message_dict.get("tool_calls"):
            message_dict["content"] = message.content or None
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, FunctionMessage):
        message_dict = {
            "role": "function",
            "content": message.content,
            "name": message.name,
        }
    elif isinstance(message, ToolMessage):
        message_dict = {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
    else:
        msg = f"Got unknown type {message}"
        raise ValueError(msg)
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


def _lc_tool_call_to_openai_tool_call(tool_call: ToolCall) -> dict[str, Any]:
    """Convert a LangChain tool call to an OpenAI tool call."""
    return {
        "id": tool_call["id"],
        "type": "function",
        "function": {
            "name": tool_call["name"],
            "arguments": tool_call["args"],
        },
    }


def _lc_invalid_tool_call_to_openai_tool_call(
    invalid_tool_call: InvalidToolCall,
) -> dict[str, Any]:
    """Convert a LangChain invalid tool call to an OpenAI tool call."""
    return {
        "id": invalid_tool_call["id"],
        "type": "function",
        "function": {
            "name": invalid_tool_call["name"],
            "arguments": invalid_tool_call["args"],
        },
    }


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: type[BaseMessageChunk]
) -> BaseMessageChunk:
    """Convert a delta response to a message chunk."""
    role = _dict.get("role")
    content = _dict.get("content") or ""
    additional_kwargs: dict[str, Any] = {}

    if tool_calls := _dict.get("tool_calls"):
        additional_kwargs["tool_calls"] = tool_calls

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    if role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
    if role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    if role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict.get("name"))
    if role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(content=content, tool_call_id=_dict.get("tool_call_id"))
    if role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)
    return default_class(content=content)  # type: ignore[call-arg]


class ChatBaseten(BaseChatModel):
    r"""Baseten chat model integration.

    Setup:
        Install ``langchain-baseten`` and set environment variable ``BASETEN_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-baseten
            export BASETEN_API_KEY="your-api-key"

    Key init args — completion params:
        model: str
            Name of Baseten model to use.
        temperature: float
            Sampling temperature.
        max_tokens: Optional[int]
            Max number of tokens to generate.

    Key init args — client params:
        baseten_api_key: Optional[SecretStr]
            Baseten API key. If not passed in will be read from env var
            ``BASETEN_API_KEY``.
        baseten_api_base: Optional[str]
            Base URL path for API requests (for Model APIs).
        model_url: Optional[str]
            Optional dedicated model URL for deployed models. If provided,
            overrides baseten_api_base. Supports /predict, /sync, or /sync/v1 endpoints.
        request_timeout: Union[float, tuple[float, float], Any, None]
            Timeout for requests to Baseten completion API.
        max_retries: int
            Maximum number of retries to make when generating.

    Instantiate:
        .. code-block:: python

            from langchain_baseten import ChatBaseten

            # Option 1: Use Model APIs with model slug (recommended)
            chat = ChatBaseten(
                model="deepseek-ai/DeepSeek-V3-0324",
                temperature=0.7,
                max_tokens=256,
                # Uses default baseten_api_base for Model APIs
            )

            # Option 2: Use dedicated model URL for deployed models
            chat = ChatBaseten(
                model="your-model-name",
                model_url="https://model-<id>.api.baseten.co/environments/production/predict",
                temperature=0.7,
                max_tokens=256,
                # model_url overrides baseten_api_base
            )

    Invoke:
        .. code-block:: python

            messages = [
                (
                    "system",
                    "You are a helpful translator. Translate the user sentence to "
                    "French.",
                ),
                ("human", "I love programming."),
            ]
            chat.invoke(messages)

        .. code-block:: python

            AIMessage(
                content="J'adore la programmation.",
                response_metadata={
                    "token_usage": {
                        "completion_tokens": 5,
                        "prompt_tokens": 31,
                        "total_tokens": 36,
                    },
                    "model_name": "deepseek-ai/DeepSeek-V3-0324",
                    "finish_reason": "stop",
                },
            )

    Stream:
        .. code-block:: python

            for chunk in chat.stream(messages):
                print(chunk.content, end="")

        .. code-block:: python

            J'adore la programmation.

    Async:
        .. code-block:: python

            await chat.ainvoke(messages)

            # stream:
            # async for chunk in chat.astream(messages):
            #     print(chunk.content, end="")

            # batch:
            # await chat.abatch([messages])

    Tool calling:
        .. code-block:: python

            from pydantic import BaseModel, Field


            class GetWeather(BaseModel):
                '''Get the current weather in a given location'''

                location: str = Field(
                    ..., description="The city and state, e.g. San Francisco, CA"
                )


            class GetPopulation(BaseModel):
                '''Get the current population in a given location'''

                location: str = Field(
                    ..., description="The city and state, e.g. San Francisco, CA"
                )


            chat_with_tools = chat.bind_tools([GetWeather, GetPopulation])
            ai_msg = chat_with_tools.invoke(
                "Which city is hotter today and which is bigger: LA or NY?"
            )
            ai_msg.tool_calls

        .. code-block:: python

            [
                {
                    "name": "GetWeather",
                    "args": {"location": "Los Angeles, CA"},
                    "id": "call_1",
                },
                {
                    "name": "GetWeather",
                    "args": {"location": "New York, NY"},
                    "id": "call_2",
                },
                {
                    "name": "GetPopulation",
                    "args": {"location": "Los Angeles, CA"},
                    "id": "call_3",
                },
                {
                    "name": "GetPopulation",
                    "args": {"location": "New York, NY"},
                    "id": "call_4",
                },
            ]

    Structured output:
        .. code-block:: python

            from typing import Optional

            from pydantic import BaseModel, Field


            class Joke(BaseModel):
                '''Joke to tell user.'''

                setup: str = Field(description="The setup of the joke")
                punchline: str = Field(description="The punchline to the joke")
                rating: Optional[int] = Field(
                    description="How funny the joke is, from 1 to 10"
                )


            structured_chat = chat.with_structured_output(Joke)
            structured_chat.invoke("Tell me a joke about cats")

        .. code-block:: python

            Joke(
                setup="Why was the cat sitting on the computer?",
                punchline="To keep an eye on the mouse!",
                rating=None,
            )

    JSON mode:
        .. code-block:: python

            json_chat = chat.bind(response_format={"type": "json_object"})
            ai_msg = json_chat.invoke(
                "Return a JSON object with key 'random_ints' and a value of 10 "
                "random ints in [0-99]"
            )
            ai_msg.content

        .. code-block:: python

            '\\n{\\n  "random_ints": [23, 87, 45, 12, 78, 34, 56, 90, 11, 67]\\n}'

    Response metadata:
        .. code-block:: python

            ai_msg = chat.invoke(messages)
            ai_msg.response_metadata

        .. code-block:: python

            {
                "token_usage": {
                    "completion_tokens": 5,
                    "prompt_tokens": 28,
                    "total_tokens": 33,
                },
                "model_name": "deepseek-ai/DeepSeek-V3-0324",
                "finish_reason": "stop",
            }
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    async_client: Any = Field(default=None, exclude=True)  #: :meta private:
    model: str = Field(alias="model_name")
    """Model name to use."""
    temperature: float = 0.7
    """What sampling temperature to use."""
    max_tokens: Optional[int] = None
    """The maximum number of tokens to generate in the completion."""
    top_p: float = 1
    """Total probability mass of tokens to consider at each step."""
    frequency_penalty: float = 0
    """Penalizes repeated tokens according to frequency."""
    presence_penalty: float = 0
    """Penalizes repeated tokens."""
    n: int = 1
    """How many completions to generate for each prompt."""
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    baseten_api_key: SecretStr = Field(
        alias="api_key",
        default_factory=secret_from_env(
            "BASETEN_API_KEY",
            error_message=(
                "You must specify an api key. "
                "You can pass it an argument as `baseten_api_key=...` or "
                "set the environment variable `BASETEN_API_KEY`."
            ),
        ),
    )
    """Automatically inferred from env var ``BASETEN_API_KEY`` if not provided."""
    baseten_api_base: Optional[str] = Field(
        alias="base_url", default="https://inference.baseten.co/v1"
    )
    """Base URL path for API requests. Leave as default for Model APIs, or provide
    dedicated model URL for dedicated deployments."""
    model_url: Optional[str] = Field(default=None)
    """Optional dedicated model URL for deployed models. If provided, this will
    override baseten_api_base. Should be in format:
    https://model-<id>.api.baseten.co/environments/production/predict or /sync/v1"""
    request_timeout: Union[float, tuple[float, float], Any, None] = Field(
        default=None, alias="timeout"
    )
    """Timeout for requests to Baseten completion API."""
    max_retries: int = 2
    """Maximum number of retries to make when generating."""
    streaming: bool = False
    """Whether to stream the results or not."""

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        return _build_model_kwargs(values, all_required_field_names)

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        try:
            import openai  # noqa: F401
        except ImportError as e:
            msg = (
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
            raise ImportError(msg) from e

        # Determine the base URL to use
        if self.model_url:
            # Use dedicated model URL, normalize for OpenAI client
            base_url = self.model_url
            if base_url.endswith("/predict"):
                # Convert /predict to /sync/v1 for OpenAI compatibility
                base_url = base_url.replace("/predict", "/sync/v1")
            elif base_url.endswith("/sync"):
                # Add /v1 for OpenAI compatibility
                base_url = f"{base_url}/v1"
            elif not base_url.endswith("/v1"):
                # Ensure it ends with /v1 for OpenAI compatibility
                if base_url.endswith("/"):
                    base_url = f"{base_url}v1"
                else:
                    base_url = f"{base_url}/v1"
        else:
            # Use general Model APIs
            base_url = self.baseten_api_base

        # Create OpenAI clients configured for Baseten
        client_params = {
            "api_key": self.baseten_api_key.get_secret_value(),
            "base_url": base_url,
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
        }

        self.client = OpenAI(**client_params)
        self.async_client = AsyncOpenAI(**client_params)
        return self

    @property
    def _default_params(self) -> dict[str, Any]:
        """Get the default parameters for calling Baseten API."""
        params = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "n": self.n,
            "stream": self.streaming,
            **self.model_kwargs,
        }
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        return params

    def _create_chat_result(self, response: Any) -> ChatResult:
        """Create a chat result from a Baseten response."""
        generations = []
        token_usage = response.usage
        for res in response.choices:
            message = _convert_dict_to_message(res.message.model_dump())
            if token_usage:
                generation_info = {
                    "finish_reason": res.finish_reason,
                    "logprobs": getattr(res, "logprobs", None),
                }
            else:
                generation_info = {"finish_reason": res.finish_reason}
            gen = ChatGeneration(
                message=message,
                generation_info=generation_info,
            )
            generations.append(gen)
        llm_output = {
            "token_usage": token_usage.model_dump() if token_usage else {},
            "model_name": self.model,
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call out to Baseten's endpoint to generate a chat result."""
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        _message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        response = self.client.chat.completions.create(**params)
        return self._create_chat_result(response)

    def _create_message_dicts(
        self, messages: list[BaseMessage], stop: Optional[list[str]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Create message dicts and params for the API call."""
        params = self._default_params
        if stop is not None:
            if "stop" in params:
                msg = "`stop` found in both the input and default params."
                raise ValueError(msg)
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        params["messages"] = message_dicts
        return message_dicts, params

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the chat model response."""
        _message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class = AIMessageChunk
        for chunk in self.client.chat.completions.create(**params):
            if not isinstance(chunk.choices, list) or len(chunk.choices) == 0:
                continue
            choice = chunk.choices[0]
            chunk_dict = choice.delta.model_dump()
            if len(chunk_dict) == 0:
                continue
            message_chunk = _convert_delta_to_message_chunk(
                chunk_dict, default_chunk_class
            )
            default_chunk_class = message_chunk.__class__
            chunk_generation_info = {}
            if choice.finish_reason is not None:
                chunk_generation_info["finish_reason"] = choice.finish_reason
            generation_chunk = ChatGenerationChunk(
                message=message_chunk, generation_info=chunk_generation_info
            )
            if run_manager:
                run_manager.on_llm_new_token(
                    generation_chunk.text, chunk=generation_chunk
                )
            yield generation_chunk

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronously call out to Baseten's endpoint to generate a chat result."""
        if self.streaming:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        _message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        response = await self.async_client.chat.completions.create(**params)
        return self._create_chat_result(response)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Asynchronously stream the chat model response."""
        _message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class = AIMessageChunk
        async for chunk in await self.async_client.chat.completions.create(**params):
            if not isinstance(chunk.choices, list) or len(chunk.choices) == 0:
                continue
            choice = chunk.choices[0]
            chunk_dict = choice.delta.model_dump()
            if len(chunk_dict) == 0:
                continue
            message_chunk = _convert_delta_to_message_chunk(
                chunk_dict, default_chunk_class
            )
            default_chunk_class = message_chunk.__class__
            chunk_generation_info = {}
            if choice.finish_reason is not None:
                chunk_generation_info["finish_reason"] = choice.finish_reason
            generation_chunk = ChatGenerationChunk(
                message=message_chunk, generation_info=chunk_generation_info
            )
            if run_manager:
                await run_manager.on_llm_new_token(
                    generation_chunk.text, chunk=generation_chunk
                )
            yield generation_chunk

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "n": self.n,
        }

    def _get_ls_params(
        self, stop: Optional[list[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get the parameters used to invoke the model."""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="baseten",
            ls_model_name=self.model,
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_max_tokens := params.get("max_tokens"):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop"):
            ls_params["ls_stop"] = ls_stop
        return ls_params

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "baseten-chat"

    def bind_tools(
        self,
        tools: list[Union[dict[str, Any], type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[dict, str, bool]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with OpenAI tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be  a dictionary, pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            tool_choice: Which tool to require the model to call.
                Must be the name of the single provided function,
                "auto" to automatically determine which function to call
                with the option to not call any function, "required" to force the
                model to call a tool,
                or a dict of the form:
                {"type": "function", "function": {"name": "my_function"}}.
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice is not None and tool_choice:
            if isinstance(tool_choice, str) and (
                tool_choice not in ("auto", "none", "required")
            ):
                tool_choice = {"type": "function", "function": {"name": tool_choice}}
            elif isinstance(tool_choice, bool):
                tool_choice = "required"
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: Union[dict, type[BaseModel]],
        *,
        method: Literal[
            "function_calling", "json_mode", "json_schema"
        ] = "function_calling",
        include_raw: bool = False,
        strict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[dict, BaseModel]]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema as a dict or a Pydantic class. If a Pydantic class
                then the output model will be an object of that class. If a dict then
                the output will be a dict. With a Pydantic class the returned attributes
                will be validated, whereas with a dict they will not be. If using a
                Pydantic class then you may also pass additional arguments to the
                Pydantic class constructor using the **kwargs parameter.
            method: The method to use for structured output. Can be "function_calling",
                "json_mode", or "json_schema".
            include_raw: If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed output
                will be returned. If an error occurs during output parsing it will be
                caught and returned as well. The final output will always be a dict with
                keys "raw", "parsed", and "parsing_error".
            strict: Whether to use strict mode for structured output
                (currently ignored).

        Returns:
            A Runnable that takes any ChatModel input and returns as output:

            If include_raw is True then a dict with keys:
                raw: BaseMessage
                parsed: Optional[_DictOrPydantic]
                parsing_error: Optional[BaseException]

            If include_raw is False then just _DictOrPydantic is returned,
            where _DictOrPydantic depends on the schema:

            If schema is a Pydantic class then _DictOrPydantic is the Pydantic class.

            If schema is a dict then _DictOrPydantic is a dict.
        """
        _ = kwargs.pop("strict", None)  # Accept but ignore strict parameter
        if kwargs:
            msg = f"Received unsupported arguments: {kwargs}"
            raise ValueError(msg)
        is_pydantic_schema = isinstance(schema, type) and is_basemodel_subclass(schema)

        if method == "function_calling":
            if schema is None:
                msg = (
                    "schema must be specified when method is 'function_calling'. "
                    "Received None."
                )
                raise ValueError(msg)
            formatted_tool = convert_to_openai_tool(schema)
            tool_name = formatted_tool["function"]["name"]
            llm = self.bind_tools([schema], tool_choice=tool_name)
            if is_pydantic_schema:
                output_parser: OutputParserLike = PydanticToolsParser(
                    tools=[schema], first_tool_only=True
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=tool_name, first_tool_only=True
                )
        elif method == "json_schema":
            if schema is None:
                msg = (
                    "schema must be specified when method is 'json_schema'. "
                    "Received None."
                )
                raise ValueError(msg)
            formatted_schema = convert_to_json_schema(schema)
            llm = self.bind(
                response_format={"type": "json_object", "schema": formatted_schema}
            )
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)
                if is_pydantic_schema
                else JsonOutputParser()
            )
        elif method == "json_mode":
            llm = self.bind(response_format={"type": "json_object"})
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)
                if is_pydantic_schema
                else JsonOutputParser()
            )
        else:
            msg = (
                f"Unrecognized method argument. Expected one of 'function_calling', "
                f"'json_schema', or 'json_mode'. Received: '{method}'"
            )
            raise ValueError(msg)

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        return llm | output_parser
