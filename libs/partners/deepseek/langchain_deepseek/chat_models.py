"""DeepSeek chat models."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from json import JSONDecodeError
from typing import Any, Literal, TypeAlias, cast

import openai
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import (
    LangSmithParams,
    LanguageModelInput,
    ModelProfile,
    ModelProfileRegistry,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils import from_env, secret_from_env
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_deepseek.data._profiles import _PROFILES

DEFAULT_API_BASE = "https://api.deepseek.com/v1"
DEFAULT_BETA_API_BASE = "https://api.deepseek.com/beta"

_DictOrPydanticClass: TypeAlias = dict[str, Any] | type[BaseModel]
_DictOrPydantic: TypeAlias = dict[str, Any] | BaseModel


_MODEL_PROFILES = cast("ModelProfileRegistry", _PROFILES)


def _get_default_model_profile(model_name: str) -> ModelProfile:
    default = _MODEL_PROFILES.get(model_name) or {}
    return default.copy()


class ChatDeepSeek(BaseChatOpenAI):
    """DeepSeek chat model integration to access models hosted in DeepSeek's API.

    Setup:
        Install `langchain-deepseek` and set environment variable `DEEPSEEK_API_KEY`.

        ```bash
        pip install -U langchain-deepseek
        export DEEPSEEK_API_KEY="your-api-key"
        ```

    Key init args — completion params:
        model:
            Name of DeepSeek model to use, e.g. `'deepseek-chat'`.
        temperature:
            Sampling temperature.
        max_tokens:
            Max number of tokens to generate.

    Key init args — client params:
        timeout:
            Timeout for requests.
        max_retries:
            Max number of retries.
        api_key:
            DeepSeek API key. If not passed in will be read from env var `DEEPSEEK_API_KEY`.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        ```python
        from langchain_deepseek import ChatDeepSeek

        model = ChatDeepSeek(
            model="...",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # api_key="...",
            # other params...
        )
        ```

    Invoke:
        ```python
        messages = [
            ("system", "You are a helpful translator. Translate the user sentence to French."),
            ("human", "I love programming."),
        ]
        model.invoke(messages)
        ```

    Stream:
        ```python
        for chunk in model.stream(messages):
            print(chunk.text, end="")
        ```
        ```python
        stream = model.stream(messages)
        full = next(stream)
        for chunk in stream:
            full += chunk
        full
        ```

    Async:
        ```python
        await model.ainvoke(messages)

        # stream:
        # async for chunk in (await model.astream(messages))

        # batch:
        # await model.abatch([messages])
        ```

    Tool calling:
        ```python
        from pydantic import BaseModel, Field


        class GetWeather(BaseModel):
            '''Get the current weather in a given location'''

            location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


        class GetPopulation(BaseModel):
            '''Get the current population in a given location'''

            location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


        model_with_tools = model.bind_tools([GetWeather, GetPopulation])
        ai_msg = model_with_tools.invoke("Which city is hotter today and which is bigger: LA or NY?")
        ai_msg.tool_calls
        ```

        See `ChatDeepSeek.bind_tools()` method for more.

    Structured output:
        ```python
        from typing import Optional

        from pydantic import BaseModel, Field


        class Joke(BaseModel):
            '''Joke to tell user.'''

            setup: str = Field(description="The setup of the joke")
            punchline: str = Field(description="The punchline to the joke")
            rating: int | None = Field(description="How funny the joke is, from 1 to 10")


        structured_model = model.with_structured_output(Joke)
        structured_model.invoke("Tell me a joke about cats")
        ```

        See `ChatDeepSeek.with_structured_output()` for more.

    Token usage:
        ```python
        ai_msg = model.invoke(messages)
        ai_msg.usage_metadata
        ```
        ```python
        {"input_tokens": 28, "output_tokens": 5, "total_tokens": 33}
        ```

    Response metadata:
        ```python
        ai_msg = model.invoke(messages)
        ai_msg.response_metadata
        ```
    """  # noqa: E501

    model_name: str = Field(alias="model")
    """The name of the model"""
    api_key: SecretStr | None = Field(
        default_factory=secret_from_env("DEEPSEEK_API_KEY", default=None),
    )
    """DeepSeek API key"""
    api_base: str = Field(
        default_factory=from_env("DEEPSEEK_API_BASE", default=DEFAULT_API_BASE),
    )
    """DeepSeek API base URL"""

    model_config = ConfigDict(populate_by_name=True)

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-deepseek"

    @property
    def lc_secrets(self) -> dict[str, str]:
        """A map of constructor argument names to secret ids."""
        return {"api_key": "DEEPSEEK_API_KEY"}

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        ls_params = super()._get_ls_params(stop=stop, **kwargs)
        ls_params["ls_provider"] = "deepseek"
        return ls_params

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate necessary environment vars and client params."""
        if self.api_base == DEFAULT_API_BASE and not (
            self.api_key and self.api_key.get_secret_value()
        ):
            msg = "If using default api base, DEEPSEEK_API_KEY must be set."
            raise ValueError(msg)
        client_params: dict = {
            k: v
            for k, v in {
                "api_key": self.api_key.get_secret_value() if self.api_key else None,
                "base_url": self.api_base,
                "timeout": self.request_timeout,
                "max_retries": self.max_retries,
                "default_headers": self.default_headers,
                "default_query": self.default_query,
            }.items()
            if v is not None
        }

        if not (self.client or None):
            sync_specific: dict = {"http_client": self.http_client}
            self.root_client = openai.OpenAI(**client_params, **sync_specific)
            self.client = self.root_client.chat.completions
        if not (self.async_client or None):
            async_specific: dict = {"http_client": self.http_async_client}
            self.root_async_client = openai.AsyncOpenAI(
                **client_params,
                **async_specific,
            )
            self.async_client = self.root_async_client.chat.completions
        return self

    @model_validator(mode="after")
    def _set_model_profile(self) -> Self:
        """Set model profile if not overridden."""
        if self.profile is None:
            self.profile = _get_default_model_profile(self.model_name)
        return self

    @staticmethod
    def _has_provider_reasoning_refs(reasoning_details: list) -> bool:
        """Check if reasoning_details contains provider-specific reasoning references.

        Some providers (via OpenRouter) return reasoning with special metadata
        fields like 'id', 'signature', 'data' that reference stored reasoning items.
        These formats require special handling (currently bypassed):
        - 'anthropic-claude-v1' (Claude) - has signature field
        - 'openai-responses-v1' (OpenAI) - has id field for stored items
        - 'xai-responses-v1' (xAI)

        Args:
            reasoning_details: List of reasoning detail dicts.

        Returns:
            True if format has provider-specific reasoning references.
        """
        if not reasoning_details:
            return False
        first = reasoning_details[0]
        if isinstance(first, dict):
            fmt = first.get("format", "")
            return fmt in (
                "anthropic-claude-v1",
                "openai-responses-v1",
                "xai-responses-v1",
            )
        return False

    @staticmethod
    def _normalize_reasoning(reasoning: Any) -> str | None:
        """Normalize reasoning content to a string.

        Providers like OpenRouter/MiniMax return reasoning as a list of dicts.
        This normalizes to a single string.

        Args:
            reasoning: Reasoning content in various formats.

        Returns:
            Normalized string or None if not set. Empty string is preserved.
        """
        if reasoning is None:
            return None
        if isinstance(reasoning, str):
            return reasoning  # Preserve empty string ""
        if isinstance(reasoning, list):
            if not reasoning:  # Empty list
                return None
            parts: list[str] = []
            for item in reasoning:
                if isinstance(item, dict):
                    # Expected format: {"type": "reasoning_content", "text": "..."}
                    # or {"type": "reasoning.summary", "summary": "..."}
                    value = item.get("text") or item.get("summary")
                    if value is None:
                        continue
                    parts.append(value if isinstance(value, str) else str(value))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(reasoning)

    @staticmethod
    def _strip_reasoning_text(reasoning_details: list) -> list | None:
        """Strip text from reasoning_details if single item; else return None.

        Single-item: strip text to avoid duplication with reasoning_content.
        Multi-item: keep full content to preserve per-item text mapping.

        Args:
            reasoning_details: List of reasoning detail dicts.

        Returns:
            Stripped list if single item, None if multiple items (keep original).
        """
        if len(reasoning_details) != 1:
            return None  # keep original with full content

        item = reasoning_details[0]
        if isinstance(item, dict):
            stripped = {k: v for k, v in item.items() if k != "text"}
            return [stripped]
        return None

    @staticmethod
    def _reconstruct_reasoning_details(
        reasoning_details: list,
        reasoning_content: str,
    ) -> list:
        """Reconstruct reasoning_details by re-attaching text if it was stripped.

        For single-item lists where text was stripped, re-attach reasoning_content.
        For multi-item lists where text was kept, return as-is.

        Args:
            reasoning_details: List of reasoning detail dicts.
            reasoning_content: The reasoning text to re-attach if needed.

        Returns:
            List with text field restored if it was stripped.
        """
        if not reasoning_details:
            return reasoning_details

        first = reasoning_details[0]
        # If text already exists, it wasn't stripped - return as-is
        if isinstance(first, dict) and "text" in first:
            return reasoning_details

        # Text was stripped (single item) - reconstruct
        if isinstance(first, dict):
            return [{**first, "text": reasoning_content}]
        return reasoning_details

    @staticmethod
    def _set_reasoning(
        message: BaseMessage,
        reasoning: str,
        *,
        reasoning_details: Any | None = None,
    ) -> None:
        """Attach reasoning content to a message.

        Args:
            message: The message to set reasoning on.
            reasoning: The reasoning content string.
            reasoning_details: Optional structured details.
        """
        message.additional_kwargs["reasoning_content"] = reasoning
        if reasoning_details is not None and isinstance(reasoning_details, list):
            # Skip reasoning_details for special OpenRouter formats (Claude/OpenAI/xAI)
            if ChatDeepSeek._has_provider_reasoning_refs(reasoning_details):
                return
            stripped = ChatDeepSeek._strip_reasoning_text(reasoning_details)
            if stripped is not None:
                # Single item - store metadata only (text is in reasoning_content)
                message.additional_kwargs["reasoning_details"] = stripped
            else:
                # Multiple items - keep full content (edge case: accept duplication)
                message.additional_kwargs["reasoning_details"] = reasoning_details

    @staticmethod
    def _get_msg_field(msg: Any, field: str) -> Any | None:
        """Get a field from a message object, checking model_extra as fallback.

        Args:
            msg: The message object from API response.
            field: The field name to retrieve.

        Returns:
            The field value or None if not found.
        """
        if hasattr(msg, field):
            value = getattr(msg, field)
            if value is not None:
                return value
        if hasattr(msg, "model_extra") and isinstance(msg.model_extra, dict):
            return msg.model_extra.get(field)
        return None

    def _extract_reasoning_from_message(self, msg: Any) -> tuple[Any, Any]:
        """Extract raw_reasoning and reasoning_details from a message.

        Checks multiple field locations for provider compatibility:
        - DeepSeek: reasoning_content
        - OpenRouter/MiniMax: reasoning_details
        - OpenRouter(legacy): reasoning

        Args:
            msg: The message object from API response.

        Returns:
            Tuple of (raw_reasoning, reasoning_details).
        """
        reasoning_details = self._get_msg_field(msg, "reasoning_details")

        # For formats with provider refs: use 'reasoning' field instead
        if reasoning_details and self._has_provider_reasoning_refs(reasoning_details):
            raw_reasoning = self._get_msg_field(msg, "reasoning")
            return raw_reasoning, reasoning_details

        raw_reasoning = self._get_msg_field(msg, "reasoning_content")
        if raw_reasoning is None:
            raw_reasoning = reasoning_details or self._get_msg_field(msg, "reasoning")

        return raw_reasoning, reasoning_details

    def _apply_reasoning(
        self,
        message: BaseMessage,
        raw_reasoning: Any,
        reasoning_details: Any | None,
    ) -> None:
        """Normalize raw reasoning and attach to message.

        Handles the common pattern of normalizing reasoning content and
        conditionally setting reasoning_details.

        Args:
            message: The message to attach reasoning to.
            raw_reasoning: Raw reasoning in various formats (string, list, etc).
            reasoning_details: Optional structured reasoning details.
        """
        reasoning = self._normalize_reasoning(raw_reasoning)
        if reasoning is not None:
            if reasoning_details is None and isinstance(raw_reasoning, list):
                reasoning_details = raw_reasoning
            self._set_reasoning(message, reasoning, reasoning_details=reasoning_details)
        elif reasoning_details is not None:
            # Skip reasoning_details for special OpenRouter formats (Claude/OpenAI/xAI)
            if self._has_provider_reasoning_refs(reasoning_details):
                return
            message.additional_kwargs["reasoning_details"] = reasoning_details

    def _build_reasoning_sequence(
        self,
        messages: list[BaseMessage],
    ) -> list[dict[str, Any]]:
        """Build a sequence of reasoning content aligned to AI messages.

        Args:
            messages: Original conversation messages.

        Returns:
            List of reasoning payloads (in AI message order) with normalized content
            and provider-specific reasoning_details.
        """
        reasoning_sequence: list[dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg, AIMessage):
                # Use `is not None` to preserve empty string reasoning_content
                # (important for deepseek official api which requires the field)
                # Check multiple field names for provider compatibility:
                # - DeepSeek: reasoning_content
                # - OpenRouter/MiniMax: reasoning_details
                # - OpenRouter(legacy): reasoning
                raw_reasoning = msg.additional_kwargs.get("reasoning_content")
                if raw_reasoning is None:
                    raw_reasoning = msg.additional_kwargs.get("reasoning_details")
                if raw_reasoning is None:
                    raw_reasoning = msg.additional_kwargs.get("reasoning")

                reasoning_details = msg.additional_kwargs.get("reasoning_details")
                reasoning_sequence.append(
                    {
                        "content": self._normalize_reasoning(raw_reasoning),
                        "details": reasoning_details,
                    },
                )
        return reasoning_sequence

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        # Convert input to access original messages with reasoning_content
        # Uses inherited _convert_input() from base class to avoid duplication
        messages = self._convert_input(input_).to_messages()

        # Build reasoning list from original AIMessages in order so we can re-attach
        # exactly what the caller provided, including empty strings.
        reasoning_sequence = iter(self._build_reasoning_sequence(messages))

        # Process payload messages
        for message in payload["messages"]:
            if message["role"] == "tool" and isinstance(message["content"], list):
                message["content"] = json.dumps(message["content"])
            elif message["role"] == "assistant":
                # Re-inject reasoning if it exists in original message.
                reasoning = next(reasoning_sequence, None)
                if reasoning is not None:
                    content = reasoning.get("content")
                    details = reasoning.get("details")
                    if content is not None:
                        message["reasoning_content"] = content
                    if details is not None and isinstance(details, list):
                        if content is not None:
                            message["reasoning_details"] = (
                                self._reconstruct_reasoning_details(details, content)
                            )
                        else:
                            message["reasoning_details"] = details
                # DeepSeek API expects assistant content to be a string, not a list.
                # Extract text blocks and join them, or use empty string if none exist.
                if isinstance(message["content"], list):
                    text_parts = [
                        block.get("text", "")
                        for block in message["content"]
                        if isinstance(block, dict) and block.get("type") == "text"
                    ]
                    message["content"] = "".join(text_parts) if text_parts else ""

        return payload

    def _create_chat_result(
        self,
        response: dict | openai.BaseModel,
        generation_info: dict | None = None,
    ) -> ChatResult:
        rtn = super()._create_chat_result(response, generation_info)

        if not isinstance(response, openai.BaseModel):
            return rtn

        for generation in rtn.generations:
            if generation.message.response_metadata is None:
                generation.message.response_metadata = {}
            generation.message.response_metadata["model_provider"] = "deepseek"

        choices = getattr(response, "choices", None)
        if choices:
            msg = choices[0].message
            raw_reasoning, reasoning_details = self._extract_reasoning_from_message(msg)
            self._apply_reasoning(
                rtn.generations[0].message, raw_reasoning, reasoning_details
            )

        return rtn

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )
        if (choices := chunk.get("choices")) and generation_chunk:
            top = choices[0]
            if isinstance(generation_chunk.message, AIMessageChunk):
                generation_chunk.message.response_metadata = {
                    **generation_chunk.message.response_metadata,
                    "model_provider": "deepseek",
                }
                delta = top.get("delta", {})
                # Check reasoning field names and normalize
                reasoning_details = delta.get("reasoning_details")
                if "reasoning_content" in delta:
                    raw_reasoning = delta.get("reasoning_content")
                elif "reasoning_details" in delta:
                    raw_reasoning = delta.get("reasoning_details")
                elif "reasoning" in delta:
                    raw_reasoning = delta.get("reasoning")
                else:
                    raw_reasoning = None
                self._apply_reasoning(
                    generation_chunk.message, raw_reasoning, reasoning_details
                )

        return generation_chunk

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        try:
            yield from super()._stream(
                messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
        except JSONDecodeError as e:
            msg = (
                "DeepSeek API returned an invalid response. "
                "Please check the API status and try again."
            )
            raise JSONDecodeError(
                msg,
                e.doc,
                e.pos,
            ) from e

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            return super()._generate(
                messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
        except JSONDecodeError as e:
            msg = (
                "DeepSeek API returned an invalid response. "
                "Please check the API status and try again."
            )
            raise JSONDecodeError(
                msg,
                e.doc,
                e.pos,
            ) from e

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: dict | str | bool | None = None,
        strict: bool | None = None,
        parallel_tool_calls: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tool-like objects to this chat model.

        Overrides parent to use beta endpoint when `strict=True`.

        Args:
            tools: A list of tool definitions to bind to this chat model.
            tool_choice: Which tool to require the model to call.
            strict: If True, uses beta API for strict schema validation.
            parallel_tool_calls: Set to `False` to disable parallel tool use.
            **kwargs: Additional parameters passed to parent `bind_tools`.

        Returns:
            A Runnable that takes same inputs as a chat model.
        """
        # If strict mode is enabled and using default API base, switch to beta endpoint
        if strict is True and self.api_base == DEFAULT_API_BASE:
            # Create a new instance with beta endpoint
            beta_model = self.model_copy(update={"api_base": DEFAULT_BETA_API_BASE})
            return beta_model.bind_tools(
                tools,
                tool_choice=tool_choice,
                strict=strict,
                parallel_tool_calls=parallel_tool_calls,
                **kwargs,
            )

        # Otherwise use parent implementation
        return super().bind_tools(
            tools,
            tool_choice=tool_choice,
            strict=strict,
            parallel_tool_calls=parallel_tool_calls,
            **kwargs,
        )

    def with_structured_output(
        self,
        schema: _DictOrPydanticClass | None = None,
        *,
        method: Literal[
            "function_calling",
            "json_mode",
            "json_schema",
        ] = "function_calling",
        include_raw: bool = False,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema. Can be passed in as:

                - An OpenAI function/tool schema,
                - A JSON Schema,
                - A `TypedDict` class,
                - Or a Pydantic class.

                If `schema` is a Pydantic class then the model output will be a
                Pydantic instance of that class, and the model-generated fields will be
                validated by the Pydantic class. Otherwise the model output will be a
                dict and will not be validated.

                See `langchain_core.utils.function_calling.convert_to_openai_tool` for
                more on how to properly specify types and descriptions of schema fields
                when specifying a Pydantic or `TypedDict` class.

            method: The method for steering model generation, one of:

                - `'function_calling'`:
                    Uses DeepSeek's [tool-calling features](https://api-docs.deepseek.com/guides/function_calling).
                - `'json_mode'`:
                    Uses DeepSeek's [JSON mode feature](https://api-docs.deepseek.com/guides/json_mode).

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
                Whether to enable strict schema adherence when generating the function
                call. When set to `True`, DeepSeek will use the beta API endpoint
                (`https://api.deepseek.com/beta`) for strict schema validation.
                This ensures model outputs exactly match the defined schema.

                !!! note

                    DeepSeek's strict mode requires all object properties to be marked
                    as required in the schema.

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
        """
        # Some applications require that incompatible parameters (e.g., unsupported
        # methods) be handled.
        if method == "json_schema":
            method = "function_calling"

        # If strict mode is enabled and using default API base, switch to beta endpoint
        if strict is True and self.api_base == DEFAULT_API_BASE:
            # Create a new instance with beta endpoint
            beta_model = self.model_copy(update={"api_base": DEFAULT_BETA_API_BASE})
            return beta_model.with_structured_output(
                schema,
                method=method,
                include_raw=include_raw,
                strict=strict,
                **kwargs,
            )

        return super().with_structured_output(
            schema,
            method=method,
            include_raw=include_raw,
            strict=strict,
            **kwargs,
        )
