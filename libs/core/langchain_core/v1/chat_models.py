"""Chat models for conversational AI."""

from __future__ import annotations

import copy
import typing
import warnings
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from functools import cached_property
from operator import itemgetter
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    Union,
    cast,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)
from typing_extensions import override

from langchain_core.caches import BaseCache
from langchain_core.callbacks import (
    AsyncCallbackManager,
    AsyncCallbackManagerForLLMRun,
    CallbackManager,
    CallbackManagerForLLMRun,
    Callbacks,
)
from langchain_core.language_models._utils import _normalize_messages_v1
from langchain_core.language_models.base import (
    LangSmithParams,
    LanguageModelInput,
    _get_token_ids_default_method,
    _get_verbosity,
)
from langchain_core.load import dumpd
from langchain_core.messages import (
    convert_to_openai_image_block,
    get_buffer_string,
    is_data_content_block,
)
from langchain_core.messages.ai import _LC_ID_PREFIX
from langchain_core.messages.utils import (
    convert_from_v1_message,
    convert_to_messages_v1,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
)
from langchain_core.prompt_values import PromptValue
from langchain_core.rate_limiters import BaseRateLimiter
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.runnables.config import ensure_config, run_in_executor
from langchain_core.tracers._streaming import _StreamingCallbackHandler
from langchain_core.utils.function_calling import (
    convert_to_json_schema,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import TypeBaseModel, is_basemodel_subclass
from langchain_core.v1.messages import AIMessage as AIMessageV1
from langchain_core.v1.messages import AIMessageChunk as AIMessageChunkV1
from langchain_core.v1.messages import HumanMessage as HumanMessageV1
from langchain_core.v1.messages import MessageV1, add_ai_message_chunks

if TYPE_CHECKING:
    from langchain_core.output_parsers.base import OutputParserLike
    from langchain_core.runnables import Runnable, RunnableConfig
    from langchain_core.tools import BaseTool


def _generate_response_from_error(error: BaseException) -> list[AIMessageV1]:
    response = getattr(error, "response", None)
    if response is not None:
        metadata: dict = {}
        if hasattr(response, "headers"):
            try:
                metadata["headers"] = dict(response.headers)
            except Exception:
                metadata["headers"] = None
        if hasattr(response, "status_code"):
            metadata["status_code"] = response.status_code
        if hasattr(error, "request_id"):
            metadata["request_id"] = error.request_id
        # Permit response_metadata without model_name, model_provider fields
        generations = [AIMessageV1(content=[], response_metadata=metadata)]  # type: ignore[arg-type]
    else:
        generations = []

    return generations


def _format_for_tracing(messages: Sequence[MessageV1]) -> list[MessageV1]:
    """Format messages for tracing in on_chat_model_start.

    - Update image content blocks to OpenAI Chat Completions format (backward
    compatibility).
    - Add "type" key to content blocks that have a single key.

    Args:
        messages: List of messages to format.

    Returns:
        List of messages formatted for tracing.
    """
    messages_to_trace = []
    for message in messages:
        message_to_trace = message
        for idx, block in enumerate(message.content):
            # Update image content blocks to OpenAI Chat Completions format.
            if (
                block.get("type") == "image"
                and is_data_content_block(block)  # type: ignore[arg-type]  # permit unnecessary runtime check
                and block.get("source_type") != "id"
            ):
                if message_to_trace is message:
                    # Shallow copy
                    message_to_trace = copy.copy(message)
                    message_to_trace.content = list(message_to_trace.content)

                # TODO: for tracing purposes we store non-standard types (OpenAI format)
                # in message content. Consider typing these block formats.
                message_to_trace.content[idx] = convert_to_openai_image_block(block)  # type: ignore[arg-type, call-overload]
            else:
                pass
        messages_to_trace.append(message_to_trace)

    return messages_to_trace


def generate_from_stream(stream: Iterator[AIMessageChunkV1]) -> AIMessageV1:
    """Generate from a stream.

    Args:
        stream: Iterator of AIMessageChunkV1.

    Returns:
        AIMessageV1: aggregated message.
    """
    generation = next(stream, None)
    if generation:
        generation += list(stream)
    if generation is None:
        msg = "No generations found in stream."
        raise ValueError(msg)
    return generation.to_message()


async def agenerate_from_stream(
    stream: AsyncIterator[AIMessageChunkV1],
) -> AIMessageV1:
    """Async generate from a stream.

    Args:
        stream: Iterator of AIMessageChunkV1.

    Returns:
        AIMessageV1: aggregated message.
    """
    chunks = [chunk async for chunk in stream]
    return await run_in_executor(None, generate_from_stream, iter(chunks))


def _format_ls_structured_output(ls_structured_output_format: Optional[dict]) -> dict:
    if ls_structured_output_format:
        try:
            ls_structured_output_format_dict = {
                "ls_structured_output_format": {
                    "kwargs": ls_structured_output_format.get("kwargs", {}),
                    "schema": convert_to_json_schema(
                        ls_structured_output_format["schema"]
                    ),
                }
            }
        except ValueError:
            ls_structured_output_format_dict = {}
    else:
        ls_structured_output_format_dict = {}

    return ls_structured_output_format_dict


class BaseChatModel(RunnableSerializable[LanguageModelInput, AIMessageV1], ABC):
    """Base class for v1 chat models.

    Key imperative methods:
        Methods that actually call the underlying model.

        +---------------------------+----------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
        | Method                    | Input                                                          | Output                                                              | Description                                                                                      |
        +===========================+================================================================+=====================================================================+==================================================================================================+
        | `invoke`                  | str | list[dict | tuple | BaseMessage] | PromptValue           | BaseMessage                                                         | A single chat model call.                                                                        |
        +---------------------------+----------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
        | `ainvoke`                 | '''                                                            | BaseMessage                                                         | Defaults to running invoke in an async executor.                                                 |
        +---------------------------+----------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
        | `stream`                  | '''                                                            | Iterator[BaseMessageChunk]                                          | Defaults to yielding output of invoke.                                                           |
        +---------------------------+----------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
        | `astream`                 | '''                                                            | AsyncIterator[BaseMessageChunk]                                     | Defaults to yielding output of ainvoke.                                                          |
        +---------------------------+----------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
        | `astream_events`          | '''                                                            | AsyncIterator[StreamEvent]                                          | Event types: 'on_chat_model_start', 'on_chat_model_stream', 'on_chat_model_end'.                 |
        +---------------------------+----------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
        | `batch`                   | list[''']                                                      | list[BaseMessage]                                                   | Defaults to running invoke in concurrent threads.                                                |
        +---------------------------+----------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
        | `abatch`                  | list[''']                                                      | list[BaseMessage]                                                   | Defaults to running ainvoke in concurrent threads.                                               |
        +---------------------------+----------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
        | `batch_as_completed`      | list[''']                                                      | Iterator[tuple[int, Union[BaseMessage, Exception]]]                 | Defaults to running invoke in concurrent threads.                                                |
        +---------------------------+----------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
        | `abatch_as_completed`     | list[''']                                                      | AsyncIterator[tuple[int, Union[BaseMessage, Exception]]]            | Defaults to running ainvoke in concurrent threads.                                               |
        +---------------------------+----------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+

        This table provides a brief overview of the main imperative methods. Please see the base Runnable reference for full documentation.

    Key declarative methods:
        Methods for creating another Runnable using the ChatModel.

        +----------------------------------+-----------------------------------------------------------------------------------------------------------+
        | Method                           | Description                                                                                               |
        +==================================+===========================================================================================================+
        | `bind_tools`                     | Create ChatModel that can call tools.                                                                     |
        +----------------------------------+-----------------------------------------------------------------------------------------------------------+
        | `with_structured_output`         | Create wrapper that structures model output using schema.                                                 |
        +----------------------------------+-----------------------------------------------------------------------------------------------------------+
        | `with_retry`                     | Create wrapper that retries model calls on failure.                                                       |
        +----------------------------------+-----------------------------------------------------------------------------------------------------------+
        | `with_fallbacks`                 | Create wrapper that falls back to other models on failure.                                                |
        +----------------------------------+-----------------------------------------------------------------------------------------------------------+
        | `configurable_fields`            | Specify init args of the model that can be configured at runtime via the RunnableConfig.                  |
        +----------------------------------+-----------------------------------------------------------------------------------------------------------+
        | `configurable_alternatives`      | Specify alternative models which can be swapped in at runtime via the RunnableConfig.                     |
        +----------------------------------+-----------------------------------------------------------------------------------------------------------+

        This table provides a brief overview of the main declarative methods. Please see the reference for each method for full documentation.

    Creating custom chat model:
        Custom chat model implementations should inherit from this class.
        Please reference the table below for information about which
        methods and properties are required or optional for implementations.

        +----------------------------------+--------------------------------------------------------------------+-------------------+
        | Method/Property                  | Description                                                        | Required/Optional |
        +==================================+====================================================================+===================+
        | `_generate`                      | Use to generate a chat result from a prompt                        | Required          |
        +----------------------------------+--------------------------------------------------------------------+-------------------+
        | `_llm_type` (property)           | Used to uniquely identify the type of the model. Used for logging. | Required          |
        +----------------------------------+--------------------------------------------------------------------+-------------------+
        | `_identifying_params` (property) | Represent model parameterization for tracing purposes.             | Optional          |
        +----------------------------------+--------------------------------------------------------------------+-------------------+
        | `_stream`                        | Use to implement streaming                                         | Optional          |
        +----------------------------------+--------------------------------------------------------------------+-------------------+
        | `_agenerate`                     | Use to implement a native async method                             | Optional          |
        +----------------------------------+--------------------------------------------------------------------+-------------------+
        | `_astream`                       | Use to implement async version of `_stream`                        | Optional          |
        +----------------------------------+--------------------------------------------------------------------+-------------------+

        Follow the guide for more information on how to implement a custom Chat Model:
        [Guide](https://python.langchain.com/docs/how_to/custom_chat_model/).

    """  # noqa: E501

    rate_limiter: Optional[BaseRateLimiter] = Field(default=None, exclude=True)
    "An optional rate limiter to use for limiting the number of requests."

    disable_streaming: Union[bool, Literal["tool_calling"]] = False
    """Whether to disable streaming for this model.

    If streaming is bypassed, then ``stream()``/``astream()``/``astream_events()`` will
    defer to ``invoke()``/``ainvoke()``.

    - If True, will always bypass streaming case.
    - If ``'tool_calling'``, will bypass streaming case only when the model is called
      with a ``tools`` keyword argument. In other words, LangChain will automatically
      switch to non-streaming behavior (``invoke()``) only when the tools argument is
      provided. This offers the best of both worlds.
    - If False (default), will always use streaming case if available.

    The main reason for this flag is that code might be written using ``.stream()`` and
    a user may want to swap out a given model for another model whose the implementation
    does not properly support streaming.
    """

    cache: Union[BaseCache, bool, None] = Field(default=None, exclude=True)
    """Whether to cache the response.

    * If true, will use the global cache.
    * If false, will not use a cache
    * If None, will use the global cache if it's set, otherwise no cache.
    * If instance of BaseCache, will use the provided cache.

    Caching is not currently supported for streaming methods of models.
    """
    verbose: bool = Field(default_factory=_get_verbosity, exclude=True, repr=False)
    """Whether to print out response text."""
    callbacks: Callbacks = Field(default=None, exclude=True)
    """Callbacks to add to the run trace."""
    tags: Optional[list[str]] = Field(default=None, exclude=True)
    """Tags to add to the run trace."""
    metadata: Optional[dict[str, Any]] = Field(default=None, exclude=True)
    """Metadata to add to the run trace."""
    custom_get_token_ids: Optional[Callable[[str], list[int]]] = Field(
        default=None, exclude=True
    )
    """Optional encoder to use for counting tokens."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @cached_property
    def _serialized(self) -> dict[str, Any]:
        return dumpd(self)

    # --- Runnable methods ---

    @field_validator("verbose", mode="before")
    def set_verbose(cls, verbose: Optional[bool]) -> bool:  # noqa: FBT001
        """If verbose is None, set it.

        This allows users to pass in None as verbose to access the global setting.

        Args:
            verbose: The verbosity setting to use.

        Returns:
            The verbosity setting to use.
        """
        if verbose is None:
            return _get_verbosity()
        return verbose

    @property
    @override
    def InputType(self) -> Any:
        """Get the input type for this runnable."""
        from langchain_core.prompt_values import (
            ChatPromptValueConcrete,
            StringPromptValue,
        )

        # This is a version of LanguageModelInput which replaces the abstract
        # base class BaseMessage with a union of its subclasses, which makes
        # for a much better schema.
        return Union[
            str,
            Union[StringPromptValue, ChatPromptValueConcrete],
            list[MessageV1],
        ]

    @property
    @override
    def OutputType(self) -> Any:
        """Get the output type for this runnable."""
        return AIMessageV1

    def _convert_input(self, model_input: LanguageModelInput) -> list[MessageV1]:
        if isinstance(model_input, PromptValue):
            return model_input.to_messages(message_version="v1")
        if isinstance(model_input, str):
            return [HumanMessageV1(content=model_input)]
        if isinstance(model_input, Sequence):
            return convert_to_messages_v1(model_input)
        msg = (
            f"Invalid input type {type(model_input)}. "
            "Must be a PromptValue, str, or list of Messages."
        )
        raise ValueError(msg)

    def _should_stream(
        self,
        *,
        async_api: bool,
        run_manager: Optional[
            Union[CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun]
        ] = None,
        **kwargs: Any,
    ) -> bool:
        """Determine if a given model call should hit the streaming API."""
        sync_not_implemented = type(self)._stream == BaseChatModel._stream  # noqa: SLF001
        async_not_implemented = type(self)._astream == BaseChatModel._astream  # noqa: SLF001

        # Check if streaming is implemented.
        if (not async_api) and sync_not_implemented:
            return False
        # Note, since async falls back to sync we check both here.
        if async_api and async_not_implemented and sync_not_implemented:
            return False

        # Check if streaming has been disabled on this instance.
        if self.disable_streaming is True:
            return False
        # We assume tools are passed in via "tools" kwarg in all models.
        if self.disable_streaming == "tool_calling" and kwargs.get("tools"):
            return False

        # Check if a runtime streaming flag has been passed in.
        if "stream" in kwargs:
            return kwargs["stream"]

        # Check if any streaming callback handlers have been passed in.
        handlers = run_manager.handlers if run_manager else []
        return any(isinstance(h, _StreamingCallbackHandler) for h in handlers)

    @override
    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AIMessageV1:
        config = ensure_config(config)
        messages = self._convert_input(input)
        ls_structured_output_format = kwargs.pop(
            "ls_structured_output_format", None
        ) or kwargs.pop("structured_output_format", None)
        ls_structured_output_format_dict = _format_ls_structured_output(
            ls_structured_output_format
        )

        params = self._get_invocation_params(**kwargs)
        options = {**kwargs, **ls_structured_output_format_dict}
        inheritable_metadata = {
            **(config.get("metadata") or {}),
            **self._get_ls_params(**kwargs),
        }
        callback_manager = CallbackManager.configure(
            config.get("callbacks"),
            self.callbacks,
            self.verbose,
            config.get("tags"),
            self.tags,
            inheritable_metadata,
            self.metadata,
        )
        (run_manager,) = callback_manager.on_chat_model_start(
            self._serialized,
            _format_for_tracing(messages),
            invocation_params=params,
            options=options,
            name=config.get("run_name"),
            run_id=config.pop("run_id", None),
            batch_size=1,
        )

        if self.rate_limiter:
            self.rate_limiter.acquire(blocking=True)

        input_messages = _normalize_messages_v1(messages)

        if self._should_stream(async_api=False, **kwargs):
            chunks: list[AIMessageChunkV1] = []
            try:
                for msg in self._stream(input_messages, **kwargs):
                    run_manager.on_llm_new_token(msg.text)
                    chunks.append(msg)
            except BaseException as e:
                run_manager.on_llm_error(e, response=_generate_response_from_error(e))
                raise
            full_message = add_ai_message_chunks(chunks[0], *chunks[1:]).to_message()
        else:
            try:
                full_message = self._invoke(input_messages, **kwargs)
            except BaseException as e:
                run_manager.on_llm_error(e)
                raise

        if run_manager and full_message.id and full_message.id.startswith("lc_"):
            full_message.id = f"{_LC_ID_PREFIX}-{run_manager.run_id}-0"

        run_manager.on_llm_end(full_message)
        return full_message

    @override
    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AIMessageV1:
        config = ensure_config(config)
        messages = self._convert_input(input)
        ls_structured_output_format = kwargs.pop(
            "ls_structured_output_format", None
        ) or kwargs.pop("structured_output_format", None)
        ls_structured_output_format_dict = _format_ls_structured_output(
            ls_structured_output_format
        )

        params = self._get_invocation_params(**kwargs)
        options = {**kwargs, **ls_structured_output_format_dict}
        inheritable_metadata = {
            **(config.get("metadata") or {}),
            **self._get_ls_params(**kwargs),
        }
        callback_manager = AsyncCallbackManager.configure(
            config.get("callbacks"),
            self.callbacks,
            self.verbose,
            config.get("tags"),
            self.tags,
            inheritable_metadata,
            self.metadata,
        )
        (run_manager,) = await callback_manager.on_chat_model_start(
            self._serialized,
            _format_for_tracing(messages),
            invocation_params=params,
            options=options,
            name=config.get("run_name"),
            run_id=config.pop("run_id", None),
            batch_size=1,
        )

        if self.rate_limiter:
            await self.rate_limiter.aacquire(blocking=True)

        # TODO: type openai image, audio, file types and permit in MessageV1
        input_messages = _normalize_messages_v1(messages)

        if self._should_stream(async_api=True, **kwargs):
            chunks: list[AIMessageChunkV1] = []
            try:
                async for msg in self._astream(input_messages, **kwargs):
                    await run_manager.on_llm_new_token(msg.text)
                    chunks.append(msg)
            except BaseException as e:
                await run_manager.on_llm_error(
                    e, response=_generate_response_from_error(e)
                )
                raise
            full_message = add_ai_message_chunks(chunks[0], *chunks[1:]).to_message()
        else:
            try:
                full_message = await self._ainvoke(input_messages, **kwargs)
            except BaseException as e:
                await run_manager.on_llm_error(
                    e, response=_generate_response_from_error(e)
                )
                raise

        if run_manager and full_message.id and full_message.id.startswith("lc_"):
            full_message.id = f"{_LC_ID_PREFIX}-{run_manager.run_id}-0"

        await run_manager.on_llm_end(full_message)
        return full_message

    @override
    def stream(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[AIMessageChunkV1]:
        if not self._should_stream(async_api=False, **{**kwargs, "stream": True}):
            # model doesn't implement streaming, so use default implementation
            yield cast(
                "AIMessageChunkV1",
                self.invoke(input, config=config, **kwargs),
            )
        else:
            config = ensure_config(config)
            messages = self._convert_input(input)
            ls_structured_output_format = kwargs.pop(
                "ls_structured_output_format", None
            ) or kwargs.pop("structured_output_format", None)
            ls_structured_output_format_dict = _format_ls_structured_output(
                ls_structured_output_format
            )

            params = self._get_invocation_params(**kwargs)
            options = {**kwargs, **ls_structured_output_format_dict}
            inheritable_metadata = {
                **(config.get("metadata") or {}),
                **self._get_ls_params(**kwargs),
            }
            callback_manager = CallbackManager.configure(
                config.get("callbacks"),
                self.callbacks,
                self.verbose,
                config.get("tags"),
                self.tags,
                inheritable_metadata,
                self.metadata,
            )
            (run_manager,) = callback_manager.on_chat_model_start(
                self._serialized,
                _format_for_tracing(messages),
                invocation_params=params,
                options=options,
                name=config.get("run_name"),
                run_id=config.pop("run_id", None),
                batch_size=1,
            )

            chunks: list[AIMessageChunkV1] = []

            if self.rate_limiter:
                self.rate_limiter.acquire(blocking=True)

            try:
                # TODO: replace this with something for new messages
                input_messages = _normalize_messages_v1(messages)
                for msg in self._stream(input_messages, **kwargs):
                    run_manager.on_llm_new_token(msg.text)
                    chunks.append(msg)
                    yield msg

                if msg.chunk_position != "last":
                    yield (AIMessageChunkV1([], chunk_position="last"))
            except BaseException as e:
                run_manager.on_llm_error(e, response=_generate_response_from_error(e))
                raise

            msg = add_ai_message_chunks(chunks[0], *chunks[1:])

            if run_manager and msg.id and msg.id.startswith("lc_"):
                msg.id = f"{_LC_ID_PREFIX}-{run_manager.run_id}-0"

            run_manager.on_llm_end(msg)

    @override
    async def astream(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunkV1]:
        if not self._should_stream(async_api=True, **{**kwargs, "stream": True}):
            # No async or sync stream is implemented, so fall back to ainvoke
            yield cast(
                "AIMessageChunkV1",
                await self.ainvoke(input, config=config, **kwargs),
            )
            return

        config = ensure_config(config)
        messages = self._convert_input(input)

        ls_structured_output_format = kwargs.pop(
            "ls_structured_output_format", None
        ) or kwargs.pop("structured_output_format", None)
        ls_structured_output_format_dict = _format_ls_structured_output(
            ls_structured_output_format
        )

        params = self._get_invocation_params(**kwargs)
        options = {**kwargs, **ls_structured_output_format_dict}
        inheritable_metadata = {
            **(config.get("metadata") or {}),
            **self._get_ls_params(**kwargs),
        }
        callback_manager = AsyncCallbackManager.configure(
            config.get("callbacks"),
            self.callbacks,
            self.verbose,
            config.get("tags"),
            self.tags,
            inheritable_metadata,
            self.metadata,
        )
        (run_manager,) = await callback_manager.on_chat_model_start(
            self._serialized,
            _format_for_tracing(messages),
            invocation_params=params,
            options=options,
            name=config.get("run_name"),
            run_id=config.pop("run_id", None),
            batch_size=1,
        )

        if self.rate_limiter:
            await self.rate_limiter.aacquire(blocking=True)

        chunks: list[AIMessageChunkV1] = []

        try:
            input_messages = _normalize_messages_v1(messages)
            async for msg in self._astream(
                input_messages,
                **kwargs,
            ):
                await run_manager.on_llm_new_token(msg.text)
                chunks.append(msg)
                yield msg
            if msg.chunk_position != "last":
                yield (AIMessageChunkV1([], chunk_position="last"))
        except BaseException as e:
            await run_manager.on_llm_error(e, response=_generate_response_from_error(e))
            raise

        msg = add_ai_message_chunks(chunks[0], *chunks[1:])

        if run_manager and msg.id and msg.id.startswith("lc_"):
            msg.id = f"{_LC_ID_PREFIX}-{run_manager.run_id}-0"

        await run_manager.on_llm_end(msg)

    # --- Custom methods ---

    def _combine_llm_outputs(self, llm_outputs: list[Optional[dict]]) -> dict:  # noqa: ARG002
        return {}

    def _get_invocation_params(
        self,
        **kwargs: Any,
    ) -> dict:
        params = self.dump()
        return {**params, **kwargs}

    def _get_ls_params(
        self,
        **kwargs: Any,
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        # get default provider from class name
        default_provider = self.__class__.__name__
        if default_provider.startswith("Chat"):
            default_provider = default_provider[4:].lower()
        elif default_provider.endswith("Chat"):
            default_provider = default_provider[:-4]
        default_provider = default_provider.lower()

        ls_params = LangSmithParams(ls_provider=default_provider, ls_model_type="chat")

        # model
        model = getattr(self, "model", None) or getattr(self, "model_name", None)
        if isinstance(model, str):
            ls_params["ls_model_name"] = model

        # temperature
        temperature = kwargs.get("temperature") or getattr(self, "temperature", None)
        if isinstance(temperature, (int, float)):
            ls_params["ls_temperature"] = temperature

        # max_tokens
        max_tokens = kwargs.get("max_tokens") or getattr(self, "max_tokens", None)
        if isinstance(max_tokens, int):
            ls_params["ls_max_tokens"] = max_tokens

        return ls_params

    def _get_llm_string(self, **kwargs: Any) -> str:
        params = self._get_invocation_params(**kwargs)
        params = {**params, **kwargs}
        return str(sorted(params.items()))

    def _invoke(
        self,
        messages: list[MessageV1],
        **kwargs: Any,
    ) -> AIMessageV1:
        raise NotImplementedError

    async def _ainvoke(
        self,
        messages: list[MessageV1],
        **kwargs: Any,
    ) -> AIMessageV1:
        return await run_in_executor(
            None,
            self._invoke,
            messages,
            **kwargs,
        )

    def _stream(
        self,
        messages: list[MessageV1],
        **kwargs: Any,
    ) -> Iterator[AIMessageChunkV1]:
        raise NotImplementedError

    async def _astream(
        self,
        messages: list[MessageV1],
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunkV1]:
        iterator = await run_in_executor(
            None,
            self._stream,
            messages,
            **kwargs,
        )
        done = object()
        while True:
            item = await run_in_executor(
                None,
                next,
                iterator,
                done,
            )
            if item is done:
                break
            yield item  # type: ignore[misc]

    @property
    @abstractmethod
    def _llm_type(self) -> str:
        """Return type of chat model."""

    def dump(self, **kwargs: Any) -> dict:  # noqa: ARG002
        """Return a dictionary of the LLM."""
        starter_dict = dict(self._identifying_params)
        starter_dict["_type"] = self._llm_type
        return starter_dict

    def bind_tools(
        self,
        tools: Sequence[
            Union[typing.Dict[str, Any], type, Callable, BaseTool]  # noqa: UP006
        ],
        *,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessageV1]:
        """Bind tools to the model.

        Args:
            tools: Sequence of tools to bind to the model.
            tool_choice: The tool to use. If "any" then any tool can be used.

        Returns:
            A Runnable that returns a message.
        """
        raise NotImplementedError

    def with_structured_output(
        self,
        schema: Union[typing.Dict, type],  # noqa: UP006
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[typing.Dict, BaseModel]]:  # noqa: UP006
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema:
                The output schema. Can be passed in as:
                    - an OpenAI function/tool schema,
                    - a JSON Schema,
                    - a TypedDict class,
                    - or a Pydantic class.
                If ``schema`` is a Pydantic class then the model output will be a
                Pydantic instance of that class, and the model-generated fields will be
                validated by the Pydantic class. Otherwise the model output will be a
                dict and will not be validated. See :meth:`langchain_core.utils.function_calling.convert_to_openai_tool`
                for more on how to properly specify types and descriptions of
                schema fields when specifying a Pydantic or TypedDict class.

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

        Example: Pydantic schema (include_raw=False):
            .. code-block:: python

                from pydantic import BaseModel

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                llm = ChatModel(model="model-name", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")

                # -> AnswerWithJustification(
                #     answer='They weigh the same',
                #     justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
                # )

        Example: Pydantic schema (include_raw=True):
            .. code-block:: python

                from pydantic import BaseModel

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                llm = ChatModel(model="model-name", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification, include_raw=True)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Ao02pnFYXD6GN1yzc0uXPsvF', 'function': {'arguments': '{"answer":"They weigh the same.","justification":"Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ."}', 'name': 'AnswerWithJustification'}, 'type': 'function'}]}),
                #     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'),
                #     'parsing_error': None
                # }

        Example: Dict schema (include_raw=False):
            .. code-block:: python

                from pydantic import BaseModel
                from langchain_core.utils.function_calling import convert_to_openai_tool

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                dict_schema = convert_to_openai_tool(AnswerWithJustification)
                llm = ChatModel(model="model-name", temperature=0)
                structured_llm = llm.with_structured_output(dict_schema)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
                # }

        .. versionchanged:: 0.2.26

                Added support for TypedDict class.
        """  # noqa: E501
        _ = kwargs.pop("method", None)
        _ = kwargs.pop("strict", None)
        if kwargs:
            msg = f"Received unsupported arguments {kwargs}"
            raise ValueError(msg)

        from langchain_core.output_parsers.openai_tools import (
            JsonOutputKeyToolsParser,
            PydanticToolsParser,
        )

        if type(self).bind_tools is BaseChatModel.bind_tools:
            msg = "with_structured_output is not implemented for this model."
            raise NotImplementedError(msg)

        llm = self.bind_tools(
            [schema],
            tool_choice="any",
            ls_structured_output_format={
                "kwargs": {"method": "function_calling"},
                "schema": schema,
            },
        )
        if isinstance(schema, type) and is_basemodel_subclass(schema):
            output_parser: OutputParserLike = PydanticToolsParser(
                tools=[cast("TypeBaseModel", schema)], first_tool_only=True
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
        return llm | output_parser

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return self.lc_attributes

    def get_token_ids(self, text: str) -> list[int]:
        """Return the ordered ids of the tokens in a text.

        Args:
            text: The string input to tokenize.

        Returns:
            A list of ids corresponding to the tokens in the text, in order they occur
                in the text.
        """
        if self.custom_get_token_ids is not None:
            return self.custom_get_token_ids(text)
        return _get_token_ids_default_method(text)

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens present in the text.

        Useful for checking if an input fits in a model's context window.

        Args:
            text: The string input to tokenize.

        Returns:
            The integer number of tokens in the text.
        """
        return len(self.get_token_ids(text))

    def get_num_tokens_from_messages(
        self,
        messages: list[MessageV1],
        tools: Optional[Sequence] = None,
    ) -> int:
        """Get the number of tokens in the messages.

        Useful for checking if an input fits in a model's context window.

        **Note**: the base implementation of get_num_tokens_from_messages ignores
        tool schemas.

        Args:
            messages: The message inputs to tokenize.
            tools: If provided, sequence of dict, BaseModel, function, or BaseTools
                to be converted to tool schemas.

        Returns:
            The sum of the number of tokens across the messages.
        """
        messages_v0 = [convert_from_v1_message(message) for message in messages]
        if tools is not None:
            warnings.warn(
                "Counting tokens in tool schemas is not yet supported. Ignoring tools.",
                stacklevel=2,
            )
        return sum(self.get_num_tokens(get_buffer_string([m])) for m in messages_v0)


def _gen_info_and_msg_metadata(
    generation: Union[ChatGeneration, ChatGenerationChunk],
) -> dict:
    return {
        **(generation.generation_info or {}),
        **generation.message.response_metadata,
    }
