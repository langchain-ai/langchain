"""Wrapper around Perplexity APIs."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator, Mapping
from operator import itemgetter
from typing import Any, Literal, TypeAlias, cast

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
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessageChunk,
)
from langchain_core.messages.ai import (
    OutputTokenDetails,
    UsageMetadata,
    subtract_usage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.utils import get_pydantic_field_names, secret_from_env
from langchain_core.utils.function_calling import convert_to_json_schema
from langchain_core.utils.pydantic import is_basemodel_subclass
from perplexity import AsyncPerplexity, Perplexity
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_perplexity.data._profiles import _PROFILES
from langchain_perplexity.output_parsers import (
    ReasoningJsonOutputParser,
    ReasoningStructuredOutputParser,
)
from langchain_perplexity.types import MediaResponse, WebSearchOptions

_DictOrPydanticClass: TypeAlias = dict[str, Any] | type[BaseModel]
_DictOrPydantic: TypeAlias = dict | BaseModel

logger = logging.getLogger(__name__)


_MODEL_PROFILES = cast("ModelProfileRegistry", _PROFILES)


def _get_default_model_profile(model_name: str) -> ModelProfile:
    default = _MODEL_PROFILES.get(model_name) or {}
    return default.copy()


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and is_basemodel_subclass(obj)


def _create_usage_metadata(token_usage: dict) -> UsageMetadata:
    """Create UsageMetadata from Perplexity token usage data.

    Args:
        token_usage: Dictionary containing token usage information from Perplexity API.

    Returns:
        UsageMetadata with properly structured token counts and details.
    """
    input_tokens = token_usage.get("prompt_tokens", 0)
    output_tokens = token_usage.get("completion_tokens", 0)
    total_tokens = token_usage.get("total_tokens", input_tokens + output_tokens)

    # Build output_token_details for Perplexity-specific fields
    output_token_details: OutputTokenDetails = {}
    if (reasoning := token_usage.get("reasoning_tokens")) is not None:
        output_token_details["reasoning"] = reasoning
    if (citation_tokens := token_usage.get("citation_tokens")) is not None:
        output_token_details["citation_tokens"] = citation_tokens  # type: ignore[typeddict-unknown-key]

    return UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        output_token_details=output_token_details,
    )


class ChatPerplexity(BaseChatModel):
    """`Perplexity AI` Chat models API.

    Setup:
        To use, you should have the environment variable `PPLX_API_KEY` set to your API key.
        Any parameters that are valid to be passed to the perplexity.create call
        can be passed in, even if not explicitly saved on this class.

        ```bash
        export PPLX_API_KEY=your_api_key
        ```

        Key init args - completion params:
            model:
                Name of the model to use. e.g. "sonar"
            temperature:
                Sampling temperature to use.
            max_tokens:
                Maximum number of tokens to generate.
            streaming:
                Whether to stream the results or not.

        Key init args - client params:
            pplx_api_key:
                API key for PerplexityChat API.
            request_timeout:
                Timeout for requests to PerplexityChat completion API.
            max_retries:
                Maximum number of retries to make when generating.

        See full list of supported init args and their descriptions in the params section.

        Instantiate:

        ```python
        from langchain_perplexity import ChatPerplexity

        model = ChatPerplexity(model="sonar", temperature=0.7)
        ```

        Invoke:

        ```python
        messages = [("system", "You are a chatbot."), ("user", "Hello!")]
        model.invoke(messages)
        ```

        Invoke with structured output:

        ```python
        from pydantic import BaseModel


        class StructuredOutput(BaseModel):
            role: str
            content: str


        model.with_structured_output(StructuredOutput)
        model.invoke(messages)
        ```

        Stream:
        ```python
        for chunk in model.stream(messages):
            print(chunk.content)
        ```

        Token usage:
        ```python
        response = model.invoke(messages)
        response.usage_metadata
        ```

        Response metadata:
        ```python
        response = model.invoke(messages)
        response.response_metadata
        ```
    """  # noqa: E501

    client: Any = Field(default=None, exclude=True)
    async_client: Any = Field(default=None, exclude=True)

    model: str = "sonar"
    """Model name."""

    temperature: float = 0.7
    """What sampling temperature to use."""

    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""

    pplx_api_key: SecretStr | None = Field(
        default_factory=secret_from_env("PPLX_API_KEY", default=None), alias="api_key"
    )
    """Perplexity API key."""

    request_timeout: float | tuple[float, float] | None = Field(None, alias="timeout")
    """Timeout for requests to PerplexityChat completion API."""

    max_retries: int = 6
    """Maximum number of retries to make when generating."""

    streaming: bool = False
    """Whether to stream the results or not."""

    max_tokens: int | None = None
    """Maximum number of tokens to generate."""

    search_mode: Literal["academic", "sec", "web"] | None = None
    """Search mode for specialized content: "academic", "sec", or "web"."""

    reasoning_effort: Literal["low", "medium", "high"] | None = None
    """Reasoning effort: "low", "medium", or "high" (default)."""

    language_preference: str | None = None
    """Language preference:"""

    search_domain_filter: list[str] | None = None
    """Search domain filter: list of domains to filter search results (max 20)."""

    return_images: bool = False
    """Whether to return images in the response."""

    return_related_questions: bool = False
    """Whether to return related questions in the response."""

    search_recency_filter: Literal["day", "week", "month", "year"] | None = None
    """Filter search results by recency: "day", "week", "month", or "year"."""

    search_after_date_filter: str | None = None
    """Search after date filter: date in format "MM/DD/YYYY" (default)."""

    search_before_date_filter: str | None = None
    """Only return results before this date (format: MM/DD/YYYY)."""

    last_updated_after_filter: str | None = None
    """Only return results updated after this date (format: MM/DD/YYYY)."""

    last_updated_before_filter: str | None = None
    """Only return results updated before this date (format: MM/DD/YYYY)."""

    disable_search: bool = False
    """Whether to disable web search entirely."""

    enable_search_classifier: bool = False
    """Whether to enable the search classifier."""

    web_search_options: WebSearchOptions | None = None
    """Configuration for web search behavior including Pro Search."""

    media_response: MediaResponse | None = None
    """Media response: "images", "videos", or "none" (default)."""

    model_config = ConfigDict(populate_by_name=True)

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"pplx_api_key": "PPLX_API_KEY"}

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                logger.warning(
                    f"""WARNING! {field_name} is not a default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )

        values["model_kwargs"] = extra
        return values

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        pplx_api_key = (
            self.pplx_api_key.get_secret_value() if self.pplx_api_key else None
        )

        if not self.client:
            self.client = Perplexity(api_key=pplx_api_key)

        if not self.async_client:
            self.async_client = AsyncPerplexity(api_key=pplx_api_key)

        return self

    @model_validator(mode="after")
    def _set_model_profile(self) -> Self:
        """Set model profile if not overridden."""
        if self.profile is None:
            self.profile = _get_default_model_profile(self.model)
        return self

    @property
    def _default_params(self) -> dict[str, Any]:
        """Get the default parameters for calling PerplexityChat API."""
        params: dict[str, Any] = {
            "max_tokens": self.max_tokens,
            "stream": self.streaming,
            "temperature": self.temperature,
        }
        if self.search_mode:
            params["search_mode"] = self.search_mode
        if self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort
        if self.language_preference:
            params["language_preference"] = self.language_preference
        if self.search_domain_filter:
            params["search_domain_filter"] = self.search_domain_filter
        if self.return_images:
            params["return_images"] = self.return_images
        if self.return_related_questions:
            params["return_related_questions"] = self.return_related_questions
        if self.search_recency_filter:
            params["search_recency_filter"] = self.search_recency_filter
        if self.search_after_date_filter:
            params["search_after_date_filter"] = self.search_after_date_filter
        if self.search_before_date_filter:
            params["search_before_date_filter"] = self.search_before_date_filter
        if self.last_updated_after_filter:
            params["last_updated_after_filter"] = self.last_updated_after_filter
        if self.last_updated_before_filter:
            params["last_updated_before_filter"] = self.last_updated_before_filter
        if self.disable_search:
            params["disable_search"] = self.disable_search
        if self.enable_search_classifier:
            params["enable_search_classifier"] = self.enable_search_classifier
        if self.web_search_options:
            params["web_search_options"] = self.web_search_options.model_dump(
                exclude_none=True
            )
        if self.media_response:
            if "extra_body" not in params:
                params["extra_body"] = {}
            params["extra_body"]["media_response"] = self.media_response.model_dump(
                exclude_none=True
            )

        return {**params, **self.model_kwargs}

    def _convert_message_to_dict(self, message: BaseMessage) -> dict[str, Any]:
        if isinstance(message, ChatMessage):
            message_dict = {"role": message.role, "content": message.content}
        elif isinstance(message, SystemMessage):
            message_dict = {"role": "system", "content": message.content}
        elif isinstance(message, HumanMessage):
            message_dict = {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            message_dict = {"role": "assistant", "content": message.content}
        else:
            raise TypeError(f"Got unknown type {message}")
        return message_dict

    def _create_message_dicts(
        self, messages: list[BaseMessage], stop: list[str] | None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        params = dict(self._invocation_params)
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [self._convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _convert_delta_to_message_chunk(
        self, _dict: Mapping[str, Any], default_class: type[BaseMessageChunk]
    ) -> BaseMessageChunk:
        role = _dict.get("role")
        content = _dict.get("content") or ""
        additional_kwargs: dict = {}
        if _dict.get("function_call"):
            function_call = dict(_dict["function_call"])
            if "name" in function_call and function_call["name"] is None:
                function_call["name"] = ""
            additional_kwargs["function_call"] = function_call
        if _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = _dict["tool_calls"]

        if role == "user" or default_class == HumanMessageChunk:
            return HumanMessageChunk(content=content)
        elif role == "assistant" or default_class == AIMessageChunk:
            return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
        elif role == "system" or default_class == SystemMessageChunk:
            return SystemMessageChunk(content=content)
        elif role == "function" or default_class == FunctionMessageChunk:
            return FunctionMessageChunk(content=content, name=_dict["name"])
        elif role == "tool" or default_class == ToolMessageChunk:
            return ToolMessageChunk(content=content, tool_call_id=_dict["tool_call_id"])
        elif role or default_class == ChatMessageChunk:
            return ChatMessageChunk(content=content, role=role)  # type: ignore[arg-type]
        else:
            return default_class(content=content)  # type: ignore[call-arg]

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        default_chunk_class = AIMessageChunk
        params.pop("stream", None)
        if stop:
            params["stop_sequences"] = stop
        stream_resp = self.client.chat.completions.create(
            messages=message_dicts, stream=True, **params
        )
        first_chunk = True
        prev_total_usage: UsageMetadata | None = None

        added_model_name: bool = False
        added_search_queries: bool = False
        for chunk in stream_resp:
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()
            # Collect standard usage metadata (transform from aggregate to delta)
            if total_usage := chunk.get("usage"):
                lc_total_usage = _create_usage_metadata(total_usage)
                if prev_total_usage:
                    usage_metadata: UsageMetadata | None = subtract_usage(
                        lc_total_usage, prev_total_usage
                    )
                else:
                    usage_metadata = lc_total_usage
                prev_total_usage = lc_total_usage
            else:
                usage_metadata = None
            if len(chunk["choices"]) == 0:
                continue
            choice = chunk["choices"][0]

            additional_kwargs = {}
            if first_chunk:
                additional_kwargs["citations"] = chunk.get("citations", [])
                for attr in ["images", "related_questions", "search_results"]:
                    if attr in chunk:
                        additional_kwargs[attr] = chunk[attr]

                if chunk.get("videos"):
                    additional_kwargs["videos"] = chunk["videos"]

                if chunk.get("reasoning_steps"):
                    additional_kwargs["reasoning_steps"] = chunk["reasoning_steps"]

            generation_info = {}
            if (model_name := chunk.get("model")) and not added_model_name:
                generation_info["model_name"] = model_name
                added_model_name = True
            # Add num_search_queries to generation_info if present
            if total_usage := chunk.get("usage"):
                if num_search_queries := total_usage.get("num_search_queries"):
                    if not added_search_queries:
                        generation_info["num_search_queries"] = num_search_queries
                        added_search_queries = True
                if search_context_size := total_usage.get("search_context_size"):
                    generation_info["search_context_size"] = search_context_size

            chunk = self._convert_delta_to_message_chunk(
                choice["delta"], default_chunk_class
            )

            if isinstance(chunk, AIMessageChunk) and usage_metadata:
                chunk.usage_metadata = usage_metadata

            if first_chunk:
                chunk.additional_kwargs |= additional_kwargs
                first_chunk = False

            if finish_reason := choice.get("finish_reason"):
                generation_info["finish_reason"] = finish_reason

            default_chunk_class = chunk.__class__
            chunk = ChatGenerationChunk(message=chunk, generation_info=generation_info)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        default_chunk_class = AIMessageChunk
        params.pop("stream", None)
        if stop:
            params["stop_sequences"] = stop
        stream_resp = await self.async_client.chat.completions.create(
            messages=message_dicts, stream=True, **params
        )
        first_chunk = True
        prev_total_usage: UsageMetadata | None = None

        added_model_name: bool = False
        added_search_queries: bool = False
        async for chunk in stream_resp:
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()
            if total_usage := chunk.get("usage"):
                lc_total_usage = _create_usage_metadata(total_usage)
                if prev_total_usage:
                    usage_metadata: UsageMetadata | None = subtract_usage(
                        lc_total_usage, prev_total_usage
                    )
                else:
                    usage_metadata = lc_total_usage
                prev_total_usage = lc_total_usage
            else:
                usage_metadata = None
            if len(chunk["choices"]) == 0:
                continue
            choice = chunk["choices"][0]

            additional_kwargs = {}
            if first_chunk:
                additional_kwargs["citations"] = chunk.get("citations", [])
                for attr in ["images", "related_questions", "search_results"]:
                    if attr in chunk:
                        additional_kwargs[attr] = chunk[attr]

                if chunk.get("videos"):
                    additional_kwargs["videos"] = chunk["videos"]

                if chunk.get("reasoning_steps"):
                    additional_kwargs["reasoning_steps"] = chunk["reasoning_steps"]

            generation_info = {}
            if (model_name := chunk.get("model")) and not added_model_name:
                generation_info["model_name"] = model_name
                added_model_name = True

            if total_usage := chunk.get("usage"):
                if num_search_queries := total_usage.get("num_search_queries"):
                    if not added_search_queries:
                        generation_info["num_search_queries"] = num_search_queries
                        added_search_queries = True
                if search_context_size := total_usage.get("search_context_size"):
                    generation_info["search_context_size"] = search_context_size

            chunk = self._convert_delta_to_message_chunk(
                choice["delta"], default_chunk_class
            )

            if isinstance(chunk, AIMessageChunk) and usage_metadata:
                chunk.usage_metadata = usage_metadata

            if first_chunk:
                chunk.additional_kwargs |= additional_kwargs
                first_chunk = False

            if finish_reason := choice.get("finish_reason"):
                generation_info["finish_reason"] = finish_reason

            default_chunk_class = chunk.__class__
            chunk = ChatGenerationChunk(message=chunk, generation_info=generation_info)
            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

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
            if stream_iter:
                return generate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        response = self.client.chat.completions.create(messages=message_dicts, **params)

        if hasattr(response, "usage") and response.usage:
            usage_dict = response.usage.model_dump()
            usage_metadata = _create_usage_metadata(usage_dict)
        else:
            usage_metadata = None
            usage_dict = {}

        additional_kwargs = {}
        for attr in ["citations", "images", "related_questions", "search_results"]:
            if hasattr(response, attr) and getattr(response, attr):
                additional_kwargs[attr] = getattr(response, attr)

        if hasattr(response, "videos") and response.videos:
            additional_kwargs["videos"] = [
                v.model_dump() if hasattr(v, "model_dump") else v
                for v in response.videos
            ]

        if hasattr(response, "reasoning_steps") and response.reasoning_steps:
            additional_kwargs["reasoning_steps"] = [
                r.model_dump() if hasattr(r, "model_dump") else r
                for r in response.reasoning_steps
            ]

        response_metadata: dict[str, Any] = {
            "model_name": getattr(response, "model", self.model)
        }
        if num_search_queries := usage_dict.get("num_search_queries"):
            response_metadata["num_search_queries"] = num_search_queries
        if search_context_size := usage_dict.get("search_context_size"):
            response_metadata["search_context_size"] = search_context_size

        message = AIMessage(
            content=response.choices[0].message.content,
            additional_kwargs=additional_kwargs,
            usage_metadata=usage_metadata,
            response_metadata=response_metadata,
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

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
            if stream_iter:
                return await agenerate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        response = await self.async_client.chat.completions.create(
            messages=message_dicts, **params
        )

        if hasattr(response, "usage") and response.usage:
            usage_dict = response.usage.model_dump()
            usage_metadata = _create_usage_metadata(usage_dict)
        else:
            usage_metadata = None
            usage_dict = {}

        additional_kwargs = {}
        for attr in ["citations", "images", "related_questions", "search_results"]:
            if hasattr(response, attr) and getattr(response, attr):
                additional_kwargs[attr] = getattr(response, attr)

        if hasattr(response, "videos") and response.videos:
            additional_kwargs["videos"] = [
                v.model_dump() if hasattr(v, "model_dump") else v
                for v in response.videos
            ]

        if hasattr(response, "reasoning_steps") and response.reasoning_steps:
            additional_kwargs["reasoning_steps"] = [
                r.model_dump() if hasattr(r, "model_dump") else r
                for r in response.reasoning_steps
            ]

        response_metadata: dict[str, Any] = {
            "model_name": getattr(response, "model", self.model)
        }
        if num_search_queries := usage_dict.get("num_search_queries"):
            response_metadata["num_search_queries"] = num_search_queries
        if search_context_size := usage_dict.get("search_context_size"):
            response_metadata["search_context_size"] = search_context_size

        message = AIMessage(
            content=response.choices[0].message.content,
            additional_kwargs=additional_kwargs,
            usage_metadata=usage_metadata,
            response_metadata=response_metadata,
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _invocation_params(self) -> Mapping[str, Any]:
        """Get the parameters used to invoke the model."""
        pplx_creds: dict[str, Any] = {"model": self.model}
        return {**pplx_creds, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "perplexitychat"

    def with_structured_output(
        self,
        schema: _DictOrPydanticClass | None = None,
        *,
        method: Literal["json_schema"] = "json_schema",
        include_raw: bool = False,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        """Model wrapper that returns outputs formatted to match the given schema for Preplexity.
        Currently, Perplexity only supports "json_schema" method for structured output
        as per their [official documentation](https://docs.perplexity.ai/guides/structured-outputs).

        Args:
            schema: The output schema. Can be passed in as:

                - a JSON Schema,
                - a `TypedDict` class,
                - or a Pydantic class

            method: The method for steering model generation, currently only support:

                - `'json_schema'`: Use the JSON Schema to parse the model output


            include_raw:
                If `False` then only the parsed structured output is returned.

                If an error occurs during model output parsing it will be raised.

                If `True` then both the raw model response (a `BaseMessage`) and the
                parsed model response will be returned.

                If an error occurs during output parsing it will be caught and returned
                as well.

                The final output is always a `dict` with keys `'raw'`, `'parsed'`, and
                `'parsing_error'`.
            strict:
                Unsupported: whether to enable strict schema adherence when generating
                the output. This parameter is included for compatibility with other
                chat models, but is currently ignored.

            kwargs: Additional keyword args aren't supported.

        Returns:
            A `Runnable` that takes same inputs as a
                `langchain_core.language_models.chat.BaseChatModel`. If `include_raw` is
                `False` and `schema` is a Pydantic class, `Runnable` outputs an instance
                of `schema` (i.e., a Pydantic object). Otherwise, if `include_raw` is
                `False` then `Runnable` outputs a `dict`.

                If `include_raw` is `True`, then `Runnable` outputs a `dict` with keys:

                - `'raw'`: `BaseMessage`
                - `'parsed'`: `None` if there was a parsing error, otherwise the type
                    depends on the `schema` as described above.
                - `'parsing_error'`: `BaseException | None`
        """  # noqa: E501
        if method in ("function_calling", "json_mode"):
            method = "json_schema"
        if method == "json_schema":
            if schema is None:
                raise ValueError(
                    "schema must be specified when method is not 'json_schema'. "
                    "Received None."
                )
            is_pydantic_schema = _is_pydantic_class(schema)
            response_format = convert_to_json_schema(schema)
            llm = self.bind(
                response_format={
                    "type": "json_schema",
                    "json_schema": {"schema": response_format},
                },
                ls_structured_output_format={
                    "kwargs": {"method": method},
                    "schema": response_format,
                },
            )
            output_parser = (
                ReasoningStructuredOutputParser(pydantic_object=schema)  # type: ignore[arg-type]
                if is_pydantic_schema
                else ReasoningJsonOutputParser()
            )
        else:
            raise ValueError(
                f"Unrecognized method argument. Expected 'json_schema' Received:\
                    '{method}'"
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
