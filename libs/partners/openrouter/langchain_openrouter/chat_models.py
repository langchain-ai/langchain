"""OpenRouter chat models."""

from __future__ import annotations

import json
import warnings
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from operator import itemgetter
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
    ChatMessage,
    ChatMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    InvalidToolCall,
    SystemMessage,
    SystemMessageChunk,
    ToolCall,
    ToolMessage,
    ToolMessageChunk,
    is_data_content_block,
)
from langchain_core.messages.ai import (
    InputTokenDetails,
    OutputTokenDetails,
    UsageMetadata,
)
from langchain_core.messages.block_translators.openai import (
    convert_to_openai_data_block,
)
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
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
from langchain_core.utils import from_env, get_pydantic_field_names, secret_from_env
from langchain_core.utils.function_calling import (
    convert_to_json_schema,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import is_basemodel_subclass
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_openrouter.data._profiles import _PROFILES

_MODEL_PROFILES = cast("ModelProfileRegistry", _PROFILES)

# LangChain-internal kwargs that must not be forwarded to the SDK.
_INTERNAL_KWARGS = frozenset({"ls_structured_output_format"})


def _get_default_model_profile(model_name: str) -> ModelProfile:
    default = _MODEL_PROFILES.get(model_name) or {}
    return default.copy()


class ChatOpenRouter(BaseChatModel):
    """OpenRouter chat model integration.

    OpenRouter is a unified API that provides access to hundreds of models from
    multiple providers (OpenAI, Anthropic, Google, Meta, etc.).

    ???+ info "Setup"

        Install `langchain-openrouter` and set environment variable
        `OPENROUTER_API_KEY`.

        ```bash
        pip install -U langchain-openrouter
        ```

        ```bash
        export OPENROUTER_API_KEY="your-api-key"
        ```

    ??? info "Key init args — completion params"

        | Param | Type | Description |
        | ----- | ---- | ----------- |
        | `model` | `str` | Model name, e.g. `'openai/gpt-4o-mini'`. |
        | `temperature` | `float | None` | Sampling temperature. |
        | `max_tokens` | `int | None` | Max tokens to generate. |

    ??? info "Key init args — client params"

        | Param | Type | Description |
        | ----- | ---- | ----------- |
        | `api_key` | `str | None` | OpenRouter API key. |
        | `base_url` | `str | None` | Base URL for API requests. |
        | `timeout` | `int | None` | Timeout in milliseconds. |
        | `app_url` | `str | None` | App URL for attribution. |
        | `app_title` | `str | None` | App title for attribution. |
        | `max_retries` | `int` | Max retries (default `2`). Set to `0` to disable. |

    ??? info "Instantiate"

        ```python
        from langchain_openrouter import ChatOpenRouter

        model = ChatOpenRouter(
            model="anthropic/claude-sonnet-4-5",
            temperature=0,
            # api_key="...",
            # openrouter_provider={"order": ["Anthropic"]},
        )
        ```

    See https://openrouter.ai/docs for platform documentation.
    """

    client: Any = Field(default=None, exclude=True)
    """Underlying SDK client (`openrouter.OpenRouter`)."""

    openrouter_api_key: SecretStr | None = Field(
        alias="api_key",
        default_factory=secret_from_env("OPENROUTER_API_KEY", default=None),
    )
    """OpenRouter API key."""

    openrouter_api_base: str | None = Field(
        default_factory=from_env("OPENROUTER_API_BASE", default=None),
        alias="base_url",
    )
    """OpenRouter API base URL. Maps to SDK `server_url`."""

    app_url: str | None = Field(
        default_factory=from_env("OPENROUTER_APP_URL", default=None),
    )
    """Application URL for OpenRouter attribution. Maps to `HTTP-Referer` header."""

    app_title: str | None = Field(
        default_factory=from_env("OPENROUTER_APP_TITLE", default=None),
    )
    """Application title for OpenRouter attribution. Maps to `X-Title` header."""

    request_timeout: int | None = Field(default=None, alias="timeout")
    """Timeout for requests in milliseconds. Maps to SDK `timeout_ms`."""

    max_retries: int = 2
    """Maximum number of retries.

    Controls the retry backoff window via the SDK's `max_elapsed_time`.

    Set to `0` to disable retries.
    """

    model_name: str = Field(alias="model")
    """The name of the model, e.g. `'anthropic/claude-sonnet-4-5'`."""

    temperature: float | None = None
    """Sampling temperature."""

    max_tokens: int | None = None
    """Maximum number of tokens to generate."""

    max_completion_tokens: int | None = None
    """Maximum number of completion tokens to generate."""

    top_p: float | None = None
    """Nucleus sampling parameter."""

    frequency_penalty: float | None = None
    """Frequency penalty for generation."""

    presence_penalty: float | None = None
    """Presence penalty for generation."""

    seed: int | None = None
    """Random seed for reproducibility."""

    stop: list[str] | str | None = Field(default=None, alias="stop_sequences")
    """Default stop sequences."""

    n: int = Field(default=1, ge=1)
    """Number of chat completions to generate for each prompt."""

    streaming: bool = False
    """Whether to stream the results or not."""

    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Any extra model parameters for the OpenRouter API."""

    reasoning: dict[str, Any] | None = None
    """Reasoning settings to pass to OpenRouter.

    Controls how many tokens the model allocates for internal chain-of-thought
    reasoning.

    Accepts an `openrouter.components.OpenResponsesReasoningConfig` or an
    equivalent dict.

    Supported keys:

    - `effort`: Controls reasoning token budget.

        Values: `'xhigh'`, `'high'`, `'medium'`, `'low'`, `'minimal'`, `'none'`.
    - `summary`: Controls verbosity of the reasoning summary returned in the
        response.

        Values: `'auto'`, `'concise'`, `'detailed'`.

    Example: `{"effort": "high", "summary": "auto"}`

    See https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
    """

    openrouter_provider: dict[str, Any] | None = None
    """Provider preferences to pass to OpenRouter.

    Example: `{"order": ["Anthropic", "OpenAI"]}`
    """

    route: str | None = None
    """Route preference for OpenRouter. E.g. `'fallback'`."""

    plugins: list[dict[str, Any]] | None = None
    """Plugins configuration for OpenRouter."""

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                msg = f"Found {field_name} supplied twice."
                raise ValueError(msg)
            if field_name not in all_required_field_names:
                warnings.warn(
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended.""",
                    stacklevel=2,
                )
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            msg = (
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )
            raise ValueError(msg)

        values["model_kwargs"] = extra
        return values

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate configuration and build the SDK client."""
        if not (self.openrouter_api_key and self.openrouter_api_key.get_secret_value()):
            msg = "OPENROUTER_API_KEY must be set."
            raise ValueError(msg)
        if self.n > 1 and self.streaming:
            msg = "n must be 1 when streaming."
            raise ValueError(msg)

        if not self.client:
            try:
                import openrouter  # noqa: PLC0415
                from openrouter.utils import (  # noqa: PLC0415
                    BackoffStrategy,
                    RetryConfig,
                )
            except ImportError as e:
                msg = (
                    "Could not import the `openrouter` Python SDK. "
                    "Please install it with: pip install openrouter"
                )
                raise ImportError(msg) from e

            client_kwargs: dict[str, Any] = {
                "api_key": self.openrouter_api_key.get_secret_value(),
            }
            if self.openrouter_api_base:
                client_kwargs["server_url"] = self.openrouter_api_base
            if self.app_url:
                client_kwargs["http_referer"] = self.app_url
            if self.app_title:
                client_kwargs["x_title"] = self.app_title
            if self.request_timeout is not None:
                client_kwargs["timeout_ms"] = self.request_timeout
            if self.max_retries > 0:
                client_kwargs["retry_config"] = RetryConfig(
                    strategy="backoff",
                    backoff=BackoffStrategy(
                        initial_interval=500,
                        max_interval=60000,
                        exponent=1.5,
                        max_elapsed_time=self.max_retries * 150_000,
                    ),
                    retry_connection_errors=True,
                )
            self.client = openrouter.OpenRouter(**client_kwargs)
        return self

    @model_validator(mode="after")
    def _set_model_profile(self) -> Self:
        """Set model profile if not overridden."""
        if self.profile is None:
            self.profile = _get_default_model_profile(self.model_name)
        return self

    #
    # Serializable class method overrides
    #
    @property
    def lc_secrets(self) -> dict[str, str]:
        """A map of constructor argument names to secret ids."""
        return {"openrouter_api_key": "OPENROUTER_API_KEY"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by LangChain."""
        return True

    #
    # BaseChatModel method overrides
    #
    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "openrouter-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "streaming": self.streaming,
            "reasoning": self.reasoning,
            "openrouter_provider": self.openrouter_provider,
            "route": self.route,
            "model_kwargs": self.model_kwargs,
        }

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="openrouter",
            ls_model_name=params.get("model", self.model_name),
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_max_tokens := params.get("max_tokens", self.max_tokens):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop", None) or self.stop:
            ls_params["ls_stop"] = ls_stop if isinstance(ls_stop, list) else [ls_stop]
        return ls_params

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        _strip_internal_kwargs(params)
        response = self.client.chat.send(messages=message_dicts, **params)
        return self._create_chat_result(response)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        _strip_internal_kwargs(params)
        response = await self.client.chat.send_async(messages=message_dicts, **params)
        return self._create_chat_result(response)

    def _stream(  # noqa: C901
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}
        _strip_internal_kwargs(params)

        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        for chunk in self.client.chat.send(messages=message_dicts, **params):
            chunk_dict = chunk.model_dump(by_alias=True)
            if not chunk_dict.get("choices"):
                if error := chunk_dict.get("error"):
                    msg = (
                        f"OpenRouter API returned an error during streaming: "
                        f"{error.get('message', str(error))} "
                        f"(code: {error.get('code', 'unknown')})"
                    )
                    raise ValueError(msg)
                continue
            choice = chunk_dict["choices"][0]
            message_chunk = _convert_chunk_to_message_chunk(
                chunk_dict, default_chunk_class
            )
            generation_info: dict[str, Any] = {}
            if finish_reason := choice.get("finish_reason"):
                generation_info["finish_reason"] = finish_reason
                # Include response-level metadata on the final chunk
                response_model = chunk_dict.get("model")
                generation_info["model_name"] = response_model or self.model_name
                if system_fingerprint := chunk_dict.get("system_fingerprint"):
                    generation_info["system_fingerprint"] = system_fingerprint
                if native_finish_reason := choice.get("native_finish_reason"):
                    generation_info["native_finish_reason"] = native_finish_reason
                if response_id := chunk_dict.get("id"):
                    generation_info["id"] = response_id
                if created := chunk_dict.get("created"):
                    generation_info["created"] = int(created)
                if object_ := chunk_dict.get("object"):
                    generation_info["object"] = object_
            logprobs = choice.get("logprobs")
            if logprobs:
                generation_info["logprobs"] = logprobs

            if generation_info:
                generation_info["model_provider"] = "openrouter"
                message_chunk = message_chunk.model_copy(
                    update={"response_metadata": generation_info}
                )

            default_chunk_class = message_chunk.__class__
            generation_chunk = ChatGenerationChunk(
                message=message_chunk, generation_info=generation_info or None
            )

            if run_manager:
                run_manager.on_llm_new_token(
                    generation_chunk.text,
                    chunk=generation_chunk,
                    logprobs=logprobs,
                )
            yield generation_chunk

    async def _astream(  # noqa: C901
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}
        _strip_internal_kwargs(params)

        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        async for chunk in await self.client.chat.send_async(
            messages=message_dicts, **params
        ):
            chunk_dict = chunk.model_dump(by_alias=True)
            if not chunk_dict.get("choices"):
                if error := chunk_dict.get("error"):
                    msg = (
                        f"OpenRouter API returned an error during streaming: "
                        f"{error.get('message', str(error))} "
                        f"(code: {error.get('code', 'unknown')})"
                    )
                    raise ValueError(msg)
                continue
            choice = chunk_dict["choices"][0]
            message_chunk = _convert_chunk_to_message_chunk(
                chunk_dict, default_chunk_class
            )
            generation_info: dict[str, Any] = {}
            if finish_reason := choice.get("finish_reason"):
                generation_info["finish_reason"] = finish_reason
                # Include response-level metadata on the final chunk
                response_model = chunk_dict.get("model")
                generation_info["model_name"] = response_model or self.model_name
                if system_fingerprint := chunk_dict.get("system_fingerprint"):
                    generation_info["system_fingerprint"] = system_fingerprint
                if native_finish_reason := choice.get("native_finish_reason"):
                    generation_info["native_finish_reason"] = native_finish_reason
                if response_id := chunk_dict.get("id"):
                    generation_info["id"] = response_id
                if created := chunk_dict.get("created"):
                    generation_info["created"] = int(created)  # UNIX timestamp
                if object_ := chunk_dict.get("object"):
                    generation_info["object"] = object_
            logprobs = choice.get("logprobs")
            if logprobs:
                generation_info["logprobs"] = logprobs

            if generation_info:
                generation_info["model_provider"] = "openrouter"
                message_chunk = message_chunk.model_copy(
                    update={"response_metadata": generation_info}
                )

            default_chunk_class = message_chunk.__class__
            generation_chunk = ChatGenerationChunk(
                message=message_chunk, generation_info=generation_info or None
            )

            if run_manager:
                await run_manager.on_llm_new_token(
                    token=generation_chunk.text,
                    chunk=generation_chunk,
                    logprobs=logprobs,
                )
            yield generation_chunk

    #
    # Internal methods
    #
    @property
    def _default_params(self) -> dict[str, Any]:  # noqa: C901, PLR0912
        """Get the default parameters for calling OpenRouter API."""
        params: dict[str, Any] = {
            "model": self.model_name,
            "stream": self.streaming,
            **self.model_kwargs,
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.max_completion_tokens is not None:
            params["max_completion_tokens"] = self.max_completion_tokens
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            params["presence_penalty"] = self.presence_penalty
        if self.seed is not None:
            params["seed"] = self.seed
        if self.n > 1:
            params["n"] = self.n
        if self.stop is not None:
            params["stop"] = self.stop
        # OpenRouter-specific params
        if self.reasoning is not None:
            params["reasoning"] = self.reasoning
        if self.openrouter_provider is not None:
            params["provider"] = self.openrouter_provider
        if self.route is not None:
            params["route"] = self.route
        if self.plugins is not None:
            params["plugins"] = self.plugins
        return params

    def _create_message_dicts(
        self, messages: list[BaseMessage], stop: list[str] | None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        params = self._default_params
        if stop is not None:
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _create_chat_result(self, response: Any) -> ChatResult:  # noqa: C901
        """Create a `ChatResult` from an OpenRouter SDK response."""
        if not isinstance(response, dict):
            response = response.model_dump(by_alias=True)

        if error := response.get("error"):
            msg = (
                f"OpenRouter API returned an error: "
                f"{error.get('message', str(error))} "
                f"(code: {error.get('code', 'unknown')})"
            )
            raise ValueError(msg)

        generations = []
        token_usage = response.get("usage") or {}

        choices = response.get("choices", [])
        if not choices:
            msg = (
                "OpenRouter API returned a response with no choices. "
                "This may indicate a problem with the request or model availability."
            )
            raise ValueError(msg)

        # Extract top-level response metadata
        response_model = response.get("model")
        system_fingerprint = response.get("system_fingerprint")

        for res in choices:
            message = _convert_dict_to_message(res["message"])
            if token_usage and isinstance(message, AIMessage):
                message.usage_metadata = _create_usage_metadata(token_usage)
            if isinstance(message, AIMessage):
                if system_fingerprint:
                    message.response_metadata["system_fingerprint"] = system_fingerprint
                if native_finish_reason := res.get("native_finish_reason"):
                    message.response_metadata["native_finish_reason"] = (
                        native_finish_reason
                    )
            generation_info: dict[str, Any] = {
                "finish_reason": res.get("finish_reason"),
            }
            if "logprobs" in res:
                generation_info["logprobs"] = res["logprobs"]
            gen = ChatGeneration(
                message=message,
                generation_info=generation_info,
            )
            generations.append(gen)

        llm_output: dict[str, Any] = {
            "model_name": response_model or self.model_name,
        }
        if response_id := response.get("id"):
            llm_output["id"] = response_id
        if created := response.get("created"):
            llm_output["created"] = int(created)
        if object_ := response.get("object"):
            llm_output["object"] = object_
        return ChatResult(generations=generations, llm_output=llm_output)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type[BaseModel] | Callable | BaseTool],
        *,
        tool_choice: dict | str | bool | None = None,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tool-like objects to this chat model.

        Args:
            tools: A list of tool definitions to bind to this chat model.

                Supports any tool definition handled by
                `langchain_core.utils.function_calling.convert_to_openai_tool`.
            tool_choice: Which tool to require the model to call.
            strict: If `True`, model output is guaranteed to exactly match the
                JSON Schema provided in the tool definition.

                If `None`, the `strict` argument will not be passed to
                the model.
            **kwargs: Any additional parameters.
        """
        formatted_tools = [
            convert_to_openai_tool(tool, strict=strict) for tool in tools
        ]
        if tool_choice is not None and tool_choice:
            if tool_choice == "any":
                tool_choice = "required"
            if isinstance(tool_choice, str) and (
                tool_choice not in ("auto", "none", "required")
            ):
                tool_choice = {"type": "function", "function": {"name": tool_choice}}
            if isinstance(tool_choice, bool):
                if len(tools) > 1:
                    msg = (
                        "tool_choice can only be True when there is one tool. Received "
                        f"{len(tools)} tools."
                    )
                    raise ValueError(msg)
                tool_name = formatted_tools[0]["function"]["name"]
                tool_choice = {
                    "type": "function",
                    "function": {"name": tool_name},
                }
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(  # type: ignore[override]
        self,
        schema: dict | type[BaseModel] | None = None,
        *,
        method: Literal["function_calling", "json_schema"] = "function_calling",
        include_raw: bool = False,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict | BaseModel]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema as a Pydantic class, TypedDict, JSON Schema,
                or OpenAI function schema.
            method: The method for steering model generation.
            include_raw: If `True` then both the raw model response and the
                parsed model response will be returned.
            strict: If `True`, model output is guaranteed to exactly match the
                JSON Schema provided in the schema definition.

                If `None`, the `strict` argument will not be passed to
                the model.
            **kwargs: Any additional parameters.

        Returns:
            A `Runnable` that takes same inputs as a `BaseChatModel`.
        """
        if method == "json_mode":
            warnings.warn(
                "Unrecognized structured output method 'json_mode'. "
                "Defaulting to 'json_schema' method.",
                stacklevel=2,
            )
            method = "json_schema"
        is_pydantic_schema = _is_pydantic_class(schema)
        if method == "function_calling":
            if schema is None:
                msg = (
                    "schema must be specified when method is 'function_calling'. "
                    "Received None."
                )
                raise ValueError(msg)
            formatted_tool = convert_to_openai_tool(schema)
            tool_name = formatted_tool["function"]["name"]
            llm = self.bind_tools(
                [schema],
                tool_choice=tool_name,
                strict=strict,
                ls_structured_output_format={
                    "kwargs": {"method": "function_calling", "strict": strict},
                    "schema": formatted_tool,
                },
                **kwargs,
            )
            if is_pydantic_schema:
                output_parser: OutputParserLike = PydanticToolsParser(
                    tools=[schema],  # type: ignore[list-item]
                    first_tool_only=True,  # type: ignore[list-item]
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
            json_schema = convert_to_json_schema(schema)
            schema_name = json_schema.get("title", "")
            json_schema_spec: dict[str, Any] = {
                "name": schema_name,
                "schema": json_schema,
            }
            if strict is not None:
                json_schema_spec["strict"] = strict
            response_format = {
                "type": "json_schema",
                "json_schema": json_schema_spec,
            }
            ls_format_info = {
                "kwargs": {"method": "json_schema", "strict": strict},
                "schema": json_schema,
            }
            llm = self.bind(
                response_format=response_format,
                ls_structured_output_format=ls_format_info,
                **kwargs,
            )
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[type-var, arg-type]
                if is_pydantic_schema
                else JsonOutputParser()
            )
        else:
            msg = (
                f"Unrecognized method argument. Expected one of 'function_calling' "
                f"or 'json_schema'. Received: '{method}'"
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


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and is_basemodel_subclass(obj)


def _strip_internal_kwargs(params: dict[str, Any]) -> None:
    """Remove LangChain-internal keys that the SDK does not accept."""
    for key in _INTERNAL_KWARGS:
        params.pop(key, None)


#
# Type conversion helpers
#
def _convert_video_block_to_openrouter(block: dict[str, Any]) -> dict[str, Any]:
    """Convert a LangChain video content block to OpenRouter's `video_url` format.

    Args:
        block: A LangChain `VideoContentBlock`.

    Returns:
        A dict in OpenRouter's `video_url` format.

    Raises:
        ValueError: If no video source is provided.
    """
    if "url" in block:
        return {"type": "video_url", "video_url": {"url": block["url"]}}
    if "base64" in block or block.get("source_type") == "base64":
        base64_data = block["data"] if "source_type" in block else block["base64"]
        mime_type = block.get("mime_type", "video/mp4")
        return {
            "type": "video_url",
            "video_url": {"url": f"data:{mime_type};base64,{base64_data}"},
        }
    msg = "Video block must have either 'url' or 'base64' data."
    raise ValueError(msg)


def _format_message_content(content: Any) -> Any:
    """Format message content for OpenRouter API.

    Converts LangChain data content blocks to the expected format.

    Args:
        content: The message content (string or list of content blocks).

    Returns:
        Formatted content suitable for the OpenRouter API.
    """
    if content and isinstance(content, list):
        formatted: list = []
        for block in content:
            if isinstance(block, dict) and is_data_content_block(block):
                if block.get("type") == "video":
                    formatted.append(_convert_video_block_to_openrouter(block))
                else:
                    formatted.append(convert_to_openai_data_block(block))
            else:
                formatted.append(block)
        return formatted
    return content


def _convert_message_to_dict(message: BaseMessage) -> dict[str, Any]:  # noqa: C901, PLR0912
    """Convert a LangChain message to an OpenRouter-compatible dict payload.

    Handles role mapping, multimodal content formatting, tool call
    serialization, and reasoning content preservation for multi-turn
    conversations.

    Args:
        message: The LangChain message.

    Returns:
        A dict suitable for the OpenRouter chat API `messages` parameter.
    """
    message_dict: dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {
            "role": "user",
            "content": _format_message_content(message.content),
        }
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
        # Filter out non-text blocks from list content
        if isinstance(message.content, list):
            text_blocks = [
                block
                for block in message.content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            message_dict["content"] = text_blocks or ""
        if message.tool_calls or message.invalid_tool_calls:
            message_dict["tool_calls"] = [
                _lc_tool_call_to_openrouter_tool_call(tc) for tc in message.tool_calls
            ] + [
                _lc_invalid_tool_call_to_openrouter_tool_call(tc)
                for tc in message.invalid_tool_calls
            ]
            if message_dict["content"] == "" or (
                isinstance(message_dict["content"], list)
                and not message_dict["content"]
            ):
                message_dict["content"] = None
        elif "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
            if message_dict["content"] == "" or (
                isinstance(message_dict["content"], list)
                and not message_dict["content"]
            ):
                message_dict["content"] = None
        # Preserve reasoning content for multi-turn conversations (e.g.
        # tool-calling loops). OpenRouter stores reasoning in "reasoning" and
        # optional structured details in "reasoning_details".
        if "reasoning_content" in message.additional_kwargs:
            message_dict["reasoning"] = message.additional_kwargs["reasoning_content"]
        if "reasoning_details" in message.additional_kwargs:
            message_dict["reasoning_details"] = message.additional_kwargs[
                "reasoning_details"
            ]
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, ToolMessage):
        message_dict = {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
    else:
        msg = f"Got unknown type {message}"
        raise TypeError(msg)
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:  # noqa: C901
    """Convert an OpenRouter API response message dict to a LangChain message.

    Extracts tool calls, reasoning content, and maps roles to the appropriate
    LangChain message type (`HumanMessage`, `AIMessage`, `SystemMessage`,
    `ToolMessage`, or `ChatMessage`).

    Args:
        _dict: The message dictionary from the API response.

    Returns:
        The corresponding LangChain message.
    """
    id_ = _dict.get("id")
    role = _dict.get("role")
    if role == "user":
        return HumanMessage(content=_dict.get("content", ""))
    if role == "assistant":
        content = _dict.get("content", "") or ""
        additional_kwargs: dict = {}
        if reasoning := _dict.get("reasoning"):
            additional_kwargs["reasoning_content"] = reasoning
        if reasoning_details := _dict.get("reasoning_details"):
            additional_kwargs["reasoning_details"] = reasoning_details
        tool_calls = []
        invalid_tool_calls = []
        if raw_tool_calls := _dict.get("tool_calls"):
            for raw_tool_call in raw_tool_calls:
                try:
                    tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
                except Exception as e:  # noqa: BLE001, PERF203
                    invalid_tool_calls.append(
                        make_invalid_tool_call(raw_tool_call, str(e))
                    )
        return AIMessage(
            content=content,
            id=id_,
            additional_kwargs=additional_kwargs,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
            response_metadata={"model_provider": "openrouter"},
        )
    if role == "system":
        return SystemMessage(content=_dict.get("content", ""))
    if role == "tool":
        additional_kwargs = {}
        if "name" in _dict:
            additional_kwargs["name"] = _dict["name"]
        return ToolMessage(
            content=_dict.get("content", ""),
            tool_call_id=_dict.get("tool_call_id"),
            additional_kwargs=additional_kwargs,
        )
    if role is None:
        msg = (
            f"OpenRouter response message is missing the 'role' field. "
            f"Message keys: {list(_dict.keys())}"
        )
        raise ValueError(msg)
    warnings.warn(
        f"Unrecognized message role '{role}' from OpenRouter. "
        f"Falling back to ChatMessage.",
        stacklevel=2,
    )
    return ChatMessage(content=_dict.get("content", ""), role=role)


def _convert_chunk_to_message_chunk(  # noqa: C901, PLR0911
    chunk: Mapping[str, Any], default_class: type[BaseMessageChunk]
) -> BaseMessageChunk:
    """Convert a streaming chunk dict to a LangChain message chunk.

    Args:
        chunk: The streaming chunk dictionary.
        default_class: Default message chunk class.

    Returns:
        The LangChain message chunk.
    """
    choice = chunk["choices"][0]
    _dict = choice.get("delta", {})
    role = cast("str", _dict.get("role"))
    content = cast("str", _dict.get("content") or "")
    additional_kwargs: dict = {}
    tool_call_chunks: list = []

    if raw_tool_calls := _dict.get("tool_calls"):
        for rtc in raw_tool_calls:
            try:
                tool_call_chunks.append(
                    tool_call_chunk(
                        name=rtc["function"].get("name"),
                        args=rtc["function"].get("arguments"),
                        id=rtc.get("id"),
                        index=rtc["index"],
                    )
                )
            except (KeyError, TypeError, AttributeError):  # noqa: PERF203
                warnings.warn(
                    f"Skipping malformed tool call chunk during streaming: "
                    f"unexpected structure in {rtc!r}.",
                    stacklevel=2,
                )

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    if role == "assistant" or default_class == AIMessageChunk:
        if reasoning := _dict.get("reasoning"):
            additional_kwargs["reasoning_content"] = reasoning
        if reasoning_details := _dict.get("reasoning_details"):
            additional_kwargs["reasoning_details"] = reasoning_details
        usage_metadata = None
        if usage := chunk.get("usage"):
            usage_metadata = _create_usage_metadata(usage)
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_call_chunks=tool_call_chunks,  # type: ignore[arg-type]
            usage_metadata=usage_metadata,  # type: ignore[arg-type]
            response_metadata={"model_provider": "openrouter"},
        )
    if role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    if role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(
            content=content, tool_call_id=_dict.get("tool_call_id", "")
        )
    if role:
        warnings.warn(
            f"Unrecognized streaming chunk role '{role}' from OpenRouter. "
            f"Falling back to ChatMessageChunk.",
            stacklevel=2,
        )
        return ChatMessageChunk(content=content, role=role)
    if default_class is ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role or "")
    return default_class(content=content)  # type: ignore[call-arg]


def _lc_tool_call_to_openrouter_tool_call(tool_call: ToolCall) -> dict[str, Any]:
    """Convert a LangChain ``ToolCall`` to an OpenRouter tool call dict.

    Serializes `args` (a dict) via `json.dumps`.
    """
    return {
        "type": "function",
        "id": tool_call["id"],
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["args"], ensure_ascii=False),
        },
    }


def _lc_invalid_tool_call_to_openrouter_tool_call(
    invalid_tool_call: InvalidToolCall,
) -> dict[str, Any]:
    """Convert a LangChain `InvalidToolCall` to an OpenRouter tool call dict.

    Unlike the valid variant, `args` is already a raw string (not a dict) and
    is passed through as-is.
    """
    return {
        "type": "function",
        "id": invalid_tool_call["id"],
        "function": {
            "name": invalid_tool_call["name"],
            "arguments": invalid_tool_call["args"],
        },
    }


def _create_usage_metadata(token_usage: dict[str, Any]) -> UsageMetadata:
    """Create usage metadata from OpenRouter token usage response.

    OpenRouter may return token counts as floats rather than ints, so all
    values are explicitly cast to int.

    Args:
        token_usage: Token usage dict from the API response.

    Returns:
        Usage metadata with input/output token details.
    """
    input_tokens = int(
        token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0
    )
    output_tokens = int(
        token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0
    )
    total_tokens = int(token_usage.get("total_tokens") or input_tokens + output_tokens)

    input_details_dict = (
        token_usage.get("prompt_tokens_details")
        or token_usage.get("input_tokens_details")
        or {}
    )
    output_details_dict = (
        token_usage.get("completion_tokens_details")
        or token_usage.get("output_tokens_details")
        or {}
    )

    cache_read = input_details_dict.get("cached_tokens")
    cache_creation = input_details_dict.get("cache_write_tokens")
    input_token_details: dict = {
        "cache_read": int(cache_read) if cache_read is not None else None,
        "cache_creation": int(cache_creation) if cache_creation is not None else None,
    }
    reasoning_tokens = output_details_dict.get("reasoning_tokens")
    output_token_details: dict = {
        "reasoning": int(reasoning_tokens) if reasoning_tokens is not None else None,
    }
    usage_metadata: UsageMetadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }

    filtered_input = {k: v for k, v in input_token_details.items() if v is not None}
    if filtered_input:
        usage_metadata["input_token_details"] = InputTokenDetails(**filtered_input)  # type: ignore[typeddict-item]
    filtered_output = {k: v for k, v in output_token_details.items() if v is not None}
    if filtered_output:
        usage_metadata["output_token_details"] = OutputTokenDetails(**filtered_output)  # type: ignore[typeddict-item]
    return usage_metadata
