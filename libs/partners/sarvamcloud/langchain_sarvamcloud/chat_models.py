"""Sarvam AI chat models."""

from __future__ import annotations

import json
import warnings
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from typing import Any, Literal, cast

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import (
    LanguageModelInput,
    ModelProfile,
    ModelProfileRegistry,
)
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
    HumanMessage,
    InvalidToolCall,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.output_parsers.openai_tools import (
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils import get_pydantic_field_names, secret_from_env
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.pydantic import is_basemodel_subclass
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_sarvamcloud.data._profiles import _PROFILES
from langchain_sarvamcloud.version import __version__

_MODEL_PROFILES = cast("ModelProfileRegistry", _PROFILES)


def _get_default_model_profile(model_name: str) -> ModelProfile:
    """Get the default profile for a model.

    Args:
        model_name: The model identifier.

    Returns:
        The model profile dictionary, or an empty dict if not found.
    """
    default = _MODEL_PROFILES.get(model_name) or {}
    return default.copy()


def _create_usage_metadata(usage: dict[str, Any]) -> UsageMetadata:
    """Create UsageMetadata from a token usage dict.

    Args:
        usage: Token usage dictionary from API response.

    Returns:
        UsageMetadata instance.
    """
    return UsageMetadata(
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
    )


def _convert_message_to_dict(message: BaseMessage) -> dict[str, Any]:
    """Convert a LangChain message to a Sarvam API message dict.

    Args:
        message: The LangChain message.

    Returns:
        The message dictionary.

    Raises:
        TypeError: If the message type is not supported.
    """
    message_dict: dict[str, Any]
    if isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content or None}
        if isinstance(message.content, list):
            text_blocks = [
                block
                for block in message.content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            message_dict["content"] = text_blocks[0]["text"] if text_blocks else None
        if message.tool_calls or message.invalid_tool_calls:
            message_dict["tool_calls"] = [
                _lc_tool_call_to_sarvam_tool_call(tc) for tc in message.tool_calls
            ] + [
                _lc_invalid_tool_call_to_sarvam_tool_call(tc)
                for tc in message.invalid_tool_calls
            ]
            if message_dict["content"] == "":
                message_dict["content"] = None
        elif "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
            if message_dict["content"] == "":
                message_dict["content"] = None
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, ToolMessage):
        message_dict = {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
    else:
        msg = f"Got unknown message type {type(message)}"
        raise TypeError(msg)
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


def _lc_tool_call_to_sarvam_tool_call(tool_call: ToolCall) -> dict[str, Any]:
    """Convert a LangChain ToolCall to Sarvam API format.

    Args:
        tool_call: The LangChain tool call.

    Returns:
        The Sarvam-compatible tool call dict.
    """
    return {
        "type": "function",
        "id": tool_call["id"],
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["args"]),
        },
    }


def _lc_invalid_tool_call_to_sarvam_tool_call(
    invalid_tool_call: InvalidToolCall,
) -> dict[str, Any]:
    """Convert a LangChain InvalidToolCall to Sarvam API format.

    Args:
        invalid_tool_call: The invalid tool call.

    Returns:
        The Sarvam-compatible tool call dict.
    """
    return {
        "type": "function",
        "id": invalid_tool_call["id"],
        "function": {
            "name": invalid_tool_call["name"] or "",
            "arguments": invalid_tool_call["args"] or "",
        },
    }


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """Convert a Sarvam API response dict to a LangChain message.

    Args:
        _dict: The response message dict.

    Returns:
        The LangChain BaseMessage.
    """
    id_ = _dict.get("id")
    role = _dict.get("role")
    if role == "user":
        return HumanMessage(content=_dict.get("content", ""), id=id_)
    if role == "assistant":
        content = _dict.get("content", "") or ""
        additional_kwargs: dict[str, Any] = {}
        tool_calls: list[ToolCall] = []
        invalid_tool_calls: list[InvalidToolCall] = []
        if raw_tool_calls := _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = raw_tool_calls
            for raw_tc in raw_tool_calls:
                try:
                    tool_calls.append(parse_tool_call(raw_tc, return_id=True))
                except Exception as e:  # noqa: BLE001
                    invalid_tool_calls.append(make_invalid_tool_call(raw_tc, str(e)))
        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
            id=id_,
        )
    if role == "system":
        return SystemMessage(content=_dict.get("content", ""), id=id_)
    if role == "tool":
        return ToolMessage(
            content=_dict.get("content", ""),
            tool_call_id=_dict.get("tool_call_id", ""),
            id=id_,
        )
    msg = f"Got unknown role {role}"
    raise ValueError(msg)


def _convert_chunk_to_message_chunk(
    chunk: Mapping[str, Any],
    default_class: type[BaseMessageChunk],
) -> BaseMessageChunk:
    """Convert a streaming chunk dict to a LangChain message chunk.

    Args:
        chunk: The streaming chunk from the API.
        default_class: The default message chunk class.

    Returns:
        The LangChain message chunk.
    """
    choice = chunk["choices"][0]
    _dict = choice["delta"]
    role = cast("str", _dict.get("role"))
    content = cast("str", _dict.get("content") or "")
    additional_kwargs: dict[str, Any] = {}
    if _dict.get("tool_calls"):
        additional_kwargs["tool_calls"] = _dict["tool_calls"]

    if role == "user" or default_class == AIMessageChunk:
        pass
    if role == "assistant" or default_class == AIMessageChunk:
        usage = chunk.get("usage")
        usage_metadata = _create_usage_metadata(usage) if usage else None
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            usage_metadata=usage_metadata,  # type: ignore[arg-type]
        )
    return default_class(content=content)  # type: ignore[call-arg]


class ChatSarvam(BaseChatModel):
    """Sarvam AI chat model.

    Sarvam AI is India's sovereign AI platform with a focus on Indian languages.
    Supports 22+ Indian languages and provides chat completions with tool calling
    and streaming.

    To use, set the environment variable `SARVAM_API_KEY` with your
    API subscription key from https://dashboard.sarvam.ai.

    Setup:
        Install `langchain-sarvamcloud` and set the environment variable:

        ```bash
        pip install -U langchain-sarvamcloud
        export SARVAM_API_KEY="your-api-key"
        ```

    Key init args — completion params:
        model:
            Name of the Sarvam model to use. Defaults to `sarvam-105b`.
        temperature:
            Sampling temperature. Sarvam default is `0.2`.
        max_tokens:
            Max number of tokens to generate.
        reasoning_effort:
            Cognitive depth for reasoning: `low`, `medium`, or `high`.

    Key init args — client params:
        api_subscription_key:
            Sarvam API subscription key. If not passed, reads from
            `SARVAM_API_KEY` env var.
        max_retries:
            Maximum number of retries on failure.

    Instantiate:
        ```python
        from langchain_sarvamcloud import ChatSarvam

        model = ChatSarvam(
            model="sarvam-105b",
            temperature=0.2,
        )
        ```

    Invoke:
        ```python
        messages = [
            ("system", "You are a helpful assistant."),
            ("human", "हिंदी में मेरा परिचय दो।"),
        ]
        model.invoke(messages)
        ```

    Stream:
        ```python
        for chunk in model.stream(messages):
            print(chunk.text, end="")
        ```

    Tool calling:
        ```python
        from pydantic import BaseModel, Field

        class GetWeather(BaseModel):
            '''Get the current weather in a given location'''
            location: str = Field(..., description="City and state, e.g. Mumbai, MH")

        model_with_tools = model.bind_tools([GetWeather])
        ai_msg = model_with_tools.invoke("What is the weather in Delhi?")
        ```
    """

    client: Any = Field(default=None, exclude=True)
    """Sarvam AI sync client."""

    async_client: Any = Field(default=None, exclude=True)
    """Sarvam AI async client."""

    model_name: str = Field(alias="model", default="sarvam-105b")
    """Sarvam model name. Defaults to `sarvam-105b`."""

    temperature: float = 0.2
    """Sampling temperature. Sarvam's default is `0.2`."""

    top_p: float = 1.0
    """Nucleus sampling parameter."""

    max_tokens: int | None = None
    """Maximum number of tokens to generate."""

    reasoning_effort: Literal["low", "medium", "high"] | None = None
    """Cognitive depth for reasoning. Options: `low`, `medium`, `high`.

    Controls how much reasoning effort the model applies before responding.
    Only supported on `sarvam-30b` and `sarvam-105b`.
    """

    streaming: bool = False
    """Whether to stream results."""

    max_retries: int = 2
    """Maximum number of retries on transient errors."""

    api_subscription_key: SecretStr | None = Field(
        alias="api_key",
        default_factory=secret_from_env("SARVAM_API_KEY", default=None),
    )
    """Sarvam API subscription key. Reads from `SARVAM_API_KEY` env var."""

    base_url: str = "https://api.sarvam.ai/v1"
    """Base URL for the Sarvam API."""

    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Additional model parameters passed to the API."""

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict[str, Any]) -> Any:
        """Move unknown keys into model_kwargs."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                msg = f"Found {field_name} supplied twice."
                raise ValueError(msg)
            if field_name not in all_required_field_names:
                warnings.warn(
                    f"WARNING! {field_name} is not a default parameter. "
                    f"{field_name} was transferred to model_kwargs.",
                    stacklevel=2,
                )
                extra[field_name] = values.pop(field_name)
        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            msg = (
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs`."
            )
            raise ValueError(msg)
        values["model_kwargs"] = extra
        return values

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Initialize Sarvam AI client from environment."""
        try:
            from sarvamai import AsyncSarvamAI, SarvamAI  # noqa: PLC0415
        except ImportError as exc:
            msg = (
                "Could not import sarvamai python package. "
                "Please install it with `pip install sarvamai`."
            )
            raise ImportError(msg) from exc

        key = (
            self.api_subscription_key.get_secret_value()
            if self.api_subscription_key
            else None
        )
        if not self.client:
            self.client = SarvamAI(api_subscription_key=key)
        if not self.async_client:
            self.async_client = AsyncSarvamAI(api_subscription_key=key)
        return self

    @model_validator(mode="after")
    def _set_model_profile(self) -> Self:
        """Set model profile from registry if not already set."""
        if self.profile is None:
            self.profile = _get_default_model_profile(self.model_name)
        return self

    @property
    def lc_secrets(self) -> dict[str, str]:
        """Secrets mapping for LangChain serialization."""
        return {"api_subscription_key": "SARVAM_API_KEY"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by LangChain."""
        return True

    @property
    def _llm_type(self) -> str:
        """Return the LLM type identifier."""
        return "sarvam-chat"

    def _get_ls_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for LangSmith tracing."""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="sarvamcloud",
            ls_model_name=params.get("model", self.model_name),
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_max_tokens := params.get("max_tokens", self.max_tokens):
            ls_params["ls_max_tokens"] = ls_max_tokens
        return ls_params

    @property
    def _default_params(self) -> dict[str, Any]:
        """Default parameters for the Sarvam API call."""
        params: dict[str, Any] = {
            "model": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            **self.model_kwargs,
        }
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.reasoning_effort is not None:
            params["reasoning_effort"] = self.reasoning_effort
        return params

    def _create_message_dicts(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Convert messages and build API params dict.

        Args:
            messages: LangChain messages.
            stop: Optional stop sequences.

        Returns:
            Tuple of (message_dicts, params).
        """
        params = self._default_params
        if stop:
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _create_chat_result(
        self, response: dict[str, Any] | Any, params: dict[str, Any]
    ) -> ChatResult:
        """Convert API response to a ChatResult.

        Args:
            response: The API response object or dict.
            params: The params used in the request.

        Returns:
            ChatResult with parsed generations.
        """
        if not isinstance(response, dict):
            response = response.model_dump()
        generations = []
        token_usage = response.get("usage", {})
        for res in response.get("choices", []):
            message = _convert_dict_to_message(res["message"])
            if token_usage and isinstance(message, AIMessage):
                message.usage_metadata = _create_usage_metadata(token_usage)
            gen = ChatGeneration(
                message=message,
                generation_info={"finish_reason": res.get("finish_reason")},
            )
            generations.append(gen)
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model_name,
        }
        if self.reasoning_effort:
            llm_output["reasoning_effort"] = self.reasoning_effort
        return ChatResult(generations=generations, llm_output=llm_output)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response synchronously.

        Args:
            messages: Input messages.
            stop: Optional stop sequences.
            run_manager: Optional callback manager.
            **kwargs: Additional keyword arguments passed to the API.

        Returns:
            ChatResult with the model's response.
        """
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        response = self.client.chat.completions(messages=message_dicts, **params)
        return self._create_chat_result(response, params)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response asynchronously.

        Args:
            messages: Input messages.
            stop: Optional stop sequences.
            run_manager: Optional async callback manager.
            **kwargs: Additional keyword arguments passed to the API.

        Returns:
            ChatResult with the model's response.
        """
        if self.streaming:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        response = await self.async_client.chat.completions(
            messages=message_dicts, **params
        )
        return self._create_chat_result(response, params)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat response chunks synchronously.

        Args:
            messages: Input messages.
            stop: Optional stop sequences.
            run_manager: Optional callback manager.
            **kwargs: Additional keyword arguments.

        Yields:
            ChatGenerationChunk for each streamed token.
        """
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}
        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        for chunk in self.client.chat.completions(messages=message_dicts, **params):
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()  # noqa: PLW2901
            if not chunk.get("choices"):
                continue
            choice = chunk["choices"][0]
            message_chunk = _convert_chunk_to_message_chunk(chunk, default_chunk_class)
            generation_info: dict[str, Any] = {}
            if finish_reason := choice.get("finish_reason"):
                generation_info["finish_reason"] = finish_reason
                generation_info["model_name"] = self.model_name
            if generation_info:
                message_chunk = message_chunk.model_copy(
                    update={"response_metadata": generation_info}
                )
            default_chunk_class = message_chunk.__class__
            generation_chunk = ChatGenerationChunk(
                message=message_chunk,
                generation_info=generation_info or None,
            )
            if run_manager:
                run_manager.on_llm_new_token(
                    generation_chunk.text, chunk=generation_chunk
                )
            yield generation_chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream chat response chunks asynchronously.

        Args:
            messages: Input messages.
            stop: Optional stop sequences.
            run_manager: Optional async callback manager.
            **kwargs: Additional keyword arguments.

        Yields:
            ChatGenerationChunk for each streamed token.
        """
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}
        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        async for chunk in await self.async_client.chat.completions(
            messages=message_dicts, **params
        ):
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()  # noqa: PLW2901
            if not chunk.get("choices"):
                continue
            choice = chunk["choices"][0]
            message_chunk = _convert_chunk_to_message_chunk(chunk, default_chunk_class)
            generation_info: dict[str, Any] = {}
            if finish_reason := choice.get("finish_reason"):
                generation_info["finish_reason"] = finish_reason
                generation_info["model_name"] = self.model_name
            if generation_info:
                message_chunk = message_chunk.model_copy(
                    update={"response_metadata": generation_info}
                )
            default_chunk_class = message_chunk.__class__
            generation_chunk = ChatGenerationChunk(
                message=message_chunk,
                generation_info=generation_info or None,
            )
            if run_manager:
                await run_manager.on_llm_new_token(
                    token=generation_chunk.text, chunk=generation_chunk
                )
            yield generation_chunk

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type[BaseModel] | Callable | BaseTool],
        *,
        tool_choice: dict | str | bool | None = None,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tools to this model for tool calling.

        Args:
            tools: Tools to bind. Can be dicts, Pydantic models, callables, or
                `BaseTool` instances.
            tool_choice: How to select tools. Options: `"none"`, `"auto"`,
                `"required"`, or a specific tool name dict.
            strict: Not used. Included for API compatibility.
            **kwargs: Additional keyword arguments.

        Returns:
            Runnable with tools bound.
        """
        formatted_tools = [convert_to_openai_tool(t) for t in tools]
        if tool_choice is not None:
            if isinstance(tool_choice, str) and tool_choice in ("auto", "none", "required"):
                kwargs["tool_choice"] = tool_choice
            elif isinstance(tool_choice, bool):
                kwargs["tool_choice"] = "required" if tool_choice else "none"
            elif isinstance(tool_choice, dict):
                kwargs["tool_choice"] = tool_choice
            elif isinstance(tool_choice, str):
                kwargs["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: dict | type[BaseModel],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict | BaseModel]:
        """Return a Runnable that returns structured output.

        Args:
            schema: Output schema as a Pydantic model or JSON schema dict.
            include_raw: If `True`, return both raw and parsed output.
            **kwargs: Additional keyword arguments passed to `bind_tools`.

        Returns:
            Runnable that produces structured output matching the schema.
        """
        from langchain_core.output_parsers import JsonOutputParser  # noqa: PLC0415
        from langchain_core.output_parsers.openai_tools import (  # noqa: PLC0415
            JsonOutputKeyToolsParser,
            PydanticToolsParser,
        )

        # Consume method/strict params — Sarvam uses tool-calling for structured output
        # regardless of method, so these are accepted but not differentiated.
        kwargs.pop("method", None)
        kwargs.pop("strict", None)
        if kwargs:
            msg = "Received unsupported arguments: " + ", ".join(kwargs)
            raise ValueError(msg)
        is_pydantic_schema = is_basemodel_subclass(schema) if isinstance(schema, type) else False
        llm = self.bind_tools([schema], tool_choice="required")
        if is_pydantic_schema:
            output_parser: Any = PydanticToolsParser(
                tools=[schema],  # type: ignore[list-item]
                first_tool_only=True,
            )
        else:
            key = schema.get("title") if isinstance(schema, dict) else None
            output_parser = JsonOutputKeyToolsParser(
                key_name=key,
                first_tool_only=True,
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
        return llm | output_parser


def itemgetter(key: str) -> Callable[[dict], Any]:
    """Return a callable that retrieves `key` from a dict.

    Args:
        key: The key to retrieve.

    Returns:
        A callable that extracts the value for `key`.
    """

    def _get(d: dict) -> Any:
        return d[key]

    return _get
