from __future__ import annotations

import asyncio
import inspect
import json
import uuid
import warnings
from abc import ABC, abstractmethod
from operator import itemgetter
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
)

from typing_extensions import TypedDict

from langchain_core._api import deprecated
from langchain_core.caches import BaseCache
from langchain_core.callbacks import (
    AsyncCallbackManager,
    AsyncCallbackManagerForLLMRun,
    BaseCallbackManager,
    CallbackManager,
    CallbackManagerForLLMRun,
    Callbacks,
)
from langchain_core.globals import get_llm_cache
from langchain_core.language_models.base import BaseLanguageModel, LanguageModelInput
from langchain_core.load import dumpd, dumps
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    convert_to_messages,
    message_chunk_to_message,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    LLMResult,
    RunInfo,
)
from langchain_core.prompt_values import ChatPromptValue, PromptValue, StringPromptValue
from langchain_core.pydantic_v1 import (
    BaseModel,
    Field,
    root_validator,
)
from langchain_core.rate_limiters import BaseRateLimiter
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.runnables.config import ensure_config, run_in_executor
from langchain_core.tracers._streaming import _StreamingCallbackHandler
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.pydantic import TypeBaseModel, is_basemodel_subclass

if TYPE_CHECKING:
    from langchain_core.output_parsers.base import OutputParserLike
    from langchain_core.runnables import Runnable, RunnableConfig
    from langchain_core.tools import BaseTool


class LangSmithParams(TypedDict, total=False):
    """LangSmith parameters for tracing."""

    ls_provider: str
    """Provider of the model."""
    ls_model_name: str
    """Name of the model."""
    ls_model_type: Literal["chat"]
    """Type of the model. Should be 'chat'."""
    ls_temperature: Optional[float]
    """Temperature for generation."""
    ls_max_tokens: Optional[int]
    """Max tokens for generation."""
    ls_stop: Optional[List[str]]
    """Stop words for generation."""


def generate_from_stream(stream: Iterator[ChatGenerationChunk]) -> ChatResult:
    """Generate from a stream.

    Args:
        stream: Iterator of ChatGenerationChunk.

    Returns:
        ChatResult: Chat result.
    """

    generation = next(stream, None)
    if generation:
        generation += list(stream)
    if generation is None:
        raise ValueError("No generations found in stream.")
    return ChatResult(
        generations=[
            ChatGeneration(
                message=message_chunk_to_message(generation.message),
                generation_info=generation.generation_info,
            )
        ]
    )


async def agenerate_from_stream(
    stream: AsyncIterator[ChatGenerationChunk],
) -> ChatResult:
    """Async generate from a stream.

    Args:
        stream: Iterator of ChatGenerationChunk.

    Returns:
        ChatResult: Chat result.
    """

    chunks = [chunk async for chunk in stream]
    return await run_in_executor(None, generate_from_stream, iter(chunks))


class BaseChatModel(BaseLanguageModel[BaseMessage], ABC):
    """Base class for chat models.

    Key imperative methods:
        Methods that actually call the underlying model.

        +---------------------------+----------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
        | Method                    | Input                                                          | Output                                                              | Description                                                                                      |
        +===========================+================================================================+=====================================================================+==================================================================================================+
        | `invoke`                  | str | List[dict | tuple | BaseMessage] | PromptValue           | BaseMessage                                                         | A single chat model call.                                                                        |
        +---------------------------+----------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
        | `ainvoke`                 | '''                                                            | BaseMessage                                                         | Defaults to running invoke in an async executor.                                                 |
        +---------------------------+----------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
        | `stream`                  | '''                                                            | Iterator[BaseMessageChunk]                                          | Defaults to yielding output of invoke.                                                           |
        +---------------------------+----------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
        | `astream`                 | '''                                                            | AsyncIterator[BaseMessageChunk]                                     | Defaults to yielding output of ainvoke.                                                          |
        +---------------------------+----------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
        | `astream_events`          | '''                                                            | AsyncIterator[StreamEvent]                                          | Event types: 'on_chat_model_start', 'on_chat_model_stream', 'on_chat_model_end'.                 |
        +---------------------------+----------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
        | `batch`                   | List[''']                                                      | List[BaseMessage]                                                   | Defaults to running invoke in concurrent threads.                                                |
        +---------------------------+----------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
        | `abatch`                  | List[''']                                                      | List[BaseMessage]                                                   | Defaults to running ainvoke in concurrent threads.                                               |
        +---------------------------+----------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
        | `batch_as_completed`      | List[''']                                                      | Iterator[Tuple[int, Union[BaseMessage, Exception]]]                 | Defaults to running invoke in concurrent threads.                                                |
        +---------------------------+----------------------------------------------------------------+---------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
        | `abatch_as_completed`     | List[''']                                                      | AsyncIterator[Tuple[int, Union[BaseMessage, Exception]]]            | Defaults to running ainvoke in concurrent threads.                                               |
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
        [Guide](https://python.langchain.com/v0.2/docs/how_to/custom_chat_model/).

    """  # noqa: E501

    callback_manager: Optional[BaseCallbackManager] = Field(default=None, exclude=True)
    """[DEPRECATED] Callback manager to add to the run trace."""

    rate_limiter: Optional[BaseRateLimiter] = Field(default=None, exclude=True)
    """An optional rate limiter to use for limiting the number of requests."""

    @root_validator(pre=True)
    def raise_deprecation(cls, values: Dict) -> Dict:
        """Raise deprecation warning if callback_manager is used.

        Args:
            values (Dict): Values to validate.

        Returns:
            Dict: Validated values.

        Raises:
            DeprecationWarning: If callback_manager is used.
        """
        if values.get("callback_manager") is not None:
            warnings.warn(
                "callback_manager is deprecated. Please use callbacks instead.",
                DeprecationWarning,
            )
            values["callbacks"] = values.pop("callback_manager", None)
        return values

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    # --- Runnable methods ---

    @property
    def OutputType(self) -> Any:
        """Get the output type for this runnable."""
        return AnyMessage

    def _convert_input(self, input: LanguageModelInput) -> PromptValue:
        if isinstance(input, PromptValue):
            return input
        elif isinstance(input, str):
            return StringPromptValue(text=input)
        elif isinstance(input, Sequence):
            return ChatPromptValue(messages=convert_to_messages(input))
        else:
            raise ValueError(
                f"Invalid input type {type(input)}. "
                "Must be a PromptValue, str, or list of BaseMessages."
            )

    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        config = ensure_config(config)
        return cast(
            ChatGeneration,
            self.generate_prompt(
                [self._convert_input(input)],
                stop=stop,
                callbacks=config.get("callbacks"),
                tags=config.get("tags"),
                metadata=config.get("metadata"),
                run_name=config.get("run_name"),
                run_id=config.pop("run_id", None),
                **kwargs,
            ).generations[0][0],
        ).message

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        config = ensure_config(config)
        llm_result = await self.agenerate_prompt(
            [self._convert_input(input)],
            stop=stop,
            callbacks=config.get("callbacks"),
            tags=config.get("tags"),
            metadata=config.get("metadata"),
            run_name=config.get("run_name"),
            run_id=config.pop("run_id", None),
            **kwargs,
        )
        return cast(ChatGeneration, llm_result.generations[0][0]).message

    def stream(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[BaseMessageChunk]:
        if type(self)._stream == BaseChatModel._stream:
            # model doesn't implement streaming, so use default implementation
            yield cast(
                BaseMessageChunk, self.invoke(input, config=config, stop=stop, **kwargs)
            )
        else:
            config = ensure_config(config)
            messages = self._convert_input(input).to_messages()
            params = self._get_invocation_params(stop=stop, **kwargs)
            options = {"stop": stop, **kwargs}
            inheritable_metadata = {
                **(config.get("metadata") or {}),
                **self._get_ls_params(stop=stop, **kwargs),
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
                dumpd(self),
                [messages],
                invocation_params=params,
                options=options,
                name=config.get("run_name"),
                run_id=config.pop("run_id", None),
                batch_size=1,
            )
            generation: Optional[ChatGenerationChunk] = None

            if self.rate_limiter:
                self.rate_limiter.acquire(blocking=True)

            try:
                for chunk in self._stream(messages, stop=stop, **kwargs):
                    if chunk.message.id is None:
                        chunk.message.id = f"run-{run_manager.run_id}"
                    chunk.message.response_metadata = _gen_info_and_msg_metadata(chunk)
                    run_manager.on_llm_new_token(
                        cast(str, chunk.message.content), chunk=chunk
                    )
                    yield chunk.message
                    if generation is None:
                        generation = chunk
                    else:
                        generation += chunk
                assert generation is not None
            except BaseException as e:
                run_manager.on_llm_error(
                    e,
                    response=LLMResult(
                        generations=[[generation]] if generation else []
                    ),
                )
                raise e
            else:
                run_manager.on_llm_end(LLMResult(generations=[[generation]]))

    async def astream(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[BaseMessageChunk]:
        if (
            type(self)._astream is BaseChatModel._astream
            and type(self)._stream is BaseChatModel._stream
        ):
            # No async or sync stream is implemented, so fall back to ainvoke
            yield cast(
                BaseMessageChunk,
                await self.ainvoke(input, config=config, stop=stop, **kwargs),
            )
            return

        config = ensure_config(config)
        messages = self._convert_input(input).to_messages()
        params = self._get_invocation_params(stop=stop, **kwargs)
        options = {"stop": stop, **kwargs}
        inheritable_metadata = {
            **(config.get("metadata") or {}),
            **self._get_ls_params(stop=stop, **kwargs),
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
            dumpd(self),
            [messages],
            invocation_params=params,
            options=options,
            name=config.get("run_name"),
            run_id=config.pop("run_id", None),
            batch_size=1,
        )

        if self.rate_limiter:
            self.rate_limiter.acquire(blocking=True)

        generation: Optional[ChatGenerationChunk] = None
        try:
            async for chunk in self._astream(
                messages,
                stop=stop,
                **kwargs,
            ):
                if chunk.message.id is None:
                    chunk.message.id = f"run-{run_manager.run_id}"
                chunk.message.response_metadata = _gen_info_and_msg_metadata(chunk)
                await run_manager.on_llm_new_token(
                    cast(str, chunk.message.content), chunk=chunk
                )
                yield chunk.message
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
        except BaseException as e:
            await run_manager.on_llm_error(
                e,
                response=LLMResult(generations=[[generation]] if generation else []),
            )
            raise e
        else:
            await run_manager.on_llm_end(
                LLMResult(generations=[[generation]]),
            )

    # --- Custom methods ---

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        return {}

    def _get_invocation_params(
        self,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> dict:
        params = self.dict()
        params["stop"] = stop
        return {**params, **kwargs}

    def _get_ls_params(
        self,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        ls_params = LangSmithParams(ls_model_type="chat")
        if stop:
            ls_params["ls_stop"] = stop
        return ls_params

    def _get_llm_string(self, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        if self.is_lc_serializable():
            params = {**kwargs, **{"stop": stop}}
            param_string = str(sorted([(k, v) for k, v in params.items()]))
            # This code is not super efficient as it goes back and forth between
            # json and dict.
            serialized_repr = dumpd(self)
            _cleanup_llm_representation(serialized_repr, 1)
            llm_string = json.dumps(serialized_repr, sort_keys=True)
            return llm_string + "---" + param_string
        else:
            params = self._get_invocation_params(stop=stop, **kwargs)
            params = {**params, **kwargs}
            return str(sorted([(k, v) for k, v in params.items()]))

    def generate(
        self,
        messages: List[List[BaseMessage]],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        *,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Pass a sequence of prompts to the model and return model generations.

        This method should make use of batched calls for models that expose a batched
        API.

        Use this method when you want to:
            1. take advantage of batched calls,
            2. need more output from the model than just the top generated value,
            3. are building chains that are agnostic to the underlying language model
                type (e.g., pure text completion models vs chat models).

        Args:
            messages: List of list of messages.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings.
            callbacks: Callbacks to pass through. Used for executing additional
                functionality, such as logging or streaming, throughout generation.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            An LLMResult, which contains a list of candidate Generations for each input
                prompt and additional model provider-specific output.
        """
        params = self._get_invocation_params(stop=stop, **kwargs)
        options = {"stop": stop}
        inheritable_metadata = {
            **(metadata or {}),
            **self._get_ls_params(stop=stop, **kwargs),
        }

        callback_manager = CallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose,
            tags,
            self.tags,
            inheritable_metadata,
            self.metadata,
        )
        run_managers = callback_manager.on_chat_model_start(
            dumpd(self),
            messages,
            invocation_params=params,
            options=options,
            name=run_name,
            run_id=run_id,
            batch_size=len(messages),
        )
        results = []
        for i, m in enumerate(messages):
            try:
                results.append(
                    self._generate_with_cache(
                        m,
                        stop=stop,
                        run_manager=run_managers[i] if run_managers else None,
                        **kwargs,
                    )
                )
            except BaseException as e:
                if run_managers:
                    run_managers[i].on_llm_error(e, response=LLMResult(generations=[]))
                raise e
        flattened_outputs = [
            LLMResult(generations=[res.generations], llm_output=res.llm_output)  # type: ignore[list-item]
            for res in results
        ]
        llm_output = self._combine_llm_outputs([res.llm_output for res in results])
        generations = [res.generations for res in results]
        output = LLMResult(generations=generations, llm_output=llm_output)  # type: ignore[arg-type]
        if run_managers:
            run_infos = []
            for manager, flattened_output in zip(run_managers, flattened_outputs):
                manager.on_llm_end(flattened_output)
                run_infos.append(RunInfo(run_id=manager.run_id))
            output.run = run_infos
        return output

    async def agenerate(
        self,
        messages: List[List[BaseMessage]],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        *,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Asynchronously pass a sequence of prompts to a model and return generations.

        This method should make use of batched calls for models that expose a batched
        API.

        Use this method when you want to:
            1. take advantage of batched calls,
            2. need more output from the model than just the top generated value,
            3. are building chains that are agnostic to the underlying language model
                type (e.g., pure text completion models vs chat models).

        Args:
            messages: List of list of messages.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings.
            callbacks: Callbacks to pass through. Used for executing additional
                functionality, such as logging or streaming, throughout generation.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            An LLMResult, which contains a list of candidate Generations for each input
                prompt and additional model provider-specific output.
        """
        params = self._get_invocation_params(stop=stop, **kwargs)
        options = {"stop": stop}
        inheritable_metadata = {
            **(metadata or {}),
            **self._get_ls_params(stop=stop, **kwargs),
        }

        callback_manager = AsyncCallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose,
            tags,
            self.tags,
            inheritable_metadata,
            self.metadata,
        )

        run_managers = await callback_manager.on_chat_model_start(
            dumpd(self),
            messages,
            invocation_params=params,
            options=options,
            name=run_name,
            batch_size=len(messages),
            run_id=run_id,
        )

        results = await asyncio.gather(
            *[
                self._agenerate_with_cache(
                    m,
                    stop=stop,
                    run_manager=run_managers[i] if run_managers else None,
                    **kwargs,
                )
                for i, m in enumerate(messages)
            ],
            return_exceptions=True,
        )
        exceptions = []
        for i, res in enumerate(results):
            if isinstance(res, BaseException):
                if run_managers:
                    await run_managers[i].on_llm_error(
                        res, response=LLMResult(generations=[])
                    )
                exceptions.append(res)
        if exceptions:
            if run_managers:
                await asyncio.gather(
                    *[
                        run_manager.on_llm_end(
                            LLMResult(
                                generations=[res.generations],  # type: ignore[list-item, union-attr]
                                llm_output=res.llm_output,  # type: ignore[list-item, union-attr]
                            )
                        )
                        for run_manager, res in zip(run_managers, results)
                        if not isinstance(res, Exception)
                    ]
                )
            raise exceptions[0]
        flattened_outputs = [
            LLMResult(generations=[res.generations], llm_output=res.llm_output)  # type: ignore[list-item, union-attr]
            for res in results
        ]
        llm_output = self._combine_llm_outputs([res.llm_output for res in results])  # type: ignore[union-attr]
        generations = [res.generations for res in results]  # type: ignore[union-attr]
        output = LLMResult(generations=generations, llm_output=llm_output)  # type: ignore[arg-type]
        await asyncio.gather(
            *[
                run_manager.on_llm_end(flattened_output)
                for run_manager, flattened_output in zip(
                    run_managers, flattened_outputs
                )
            ]
        )
        if run_managers:
            output.run = [
                RunInfo(run_id=run_manager.run_id) for run_manager in run_managers
            ]
        return output

    def generate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        prompt_messages = [p.to_messages() for p in prompts]
        return self.generate(prompt_messages, stop=stop, callbacks=callbacks, **kwargs)

    async def agenerate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        prompt_messages = [p.to_messages() for p in prompts]
        return await self.agenerate(
            prompt_messages, stop=stop, callbacks=callbacks, **kwargs
        )

    def _generate_with_cache(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if isinstance(self.cache, BaseCache):
            llm_cache = self.cache
        else:
            llm_cache = get_llm_cache()
        # We should check the cache unless it's explicitly set to False
        # A None cache means we should use the default global cache
        # if it's configured.
        check_cache = self.cache or self.cache is None
        if check_cache:
            if llm_cache:
                llm_string = self._get_llm_string(stop=stop, **kwargs)
                prompt = dumps(messages)
                cache_val = llm_cache.lookup(prompt, llm_string)
                if isinstance(cache_val, list):
                    return ChatResult(generations=cache_val)
            elif self.cache is None:
                pass
            else:
                raise ValueError(
                    "Asked to cache, but no cache found at `langchain.cache`."
                )

        # Apply the rate limiter after checking the cache, since
        # we usually don't want to rate limit cache lookups, but
        # we do want to rate limit API requests.
        if self.rate_limiter:
            self.rate_limiter.acquire(blocking=True)

        # If stream is not explicitly set, check if implicitly requested by
        # astream_events() or astream_log(). Bail out if _stream not implemented
        if type(self)._stream != BaseChatModel._stream and kwargs.pop(
            "stream",
            (
                next(
                    (
                        True
                        for h in run_manager.handlers
                        if isinstance(h, _StreamingCallbackHandler)
                    ),
                    False,
                )
                if run_manager
                else False
            ),
        ):
            chunks: List[ChatGenerationChunk] = []
            for chunk in self._stream(messages, stop=stop, **kwargs):
                chunk.message.response_metadata = _gen_info_and_msg_metadata(chunk)
                if run_manager:
                    if chunk.message.id is None:
                        chunk.message.id = f"run-{run_manager.run_id}"
                    run_manager.on_llm_new_token(
                        cast(str, chunk.message.content), chunk=chunk
                    )
                chunks.append(chunk)
            result = generate_from_stream(iter(chunks))
        else:
            if inspect.signature(self._generate).parameters.get("run_manager"):
                result = self._generate(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
            else:
                result = self._generate(messages, stop=stop, **kwargs)

        # Add response metadata to each generation
        for idx, generation in enumerate(result.generations):
            if run_manager and generation.message.id is None:
                generation.message.id = f"run-{run_manager.run_id}-{idx}"
            generation.message.response_metadata = _gen_info_and_msg_metadata(
                generation
            )
        if len(result.generations) == 1 and result.llm_output is not None:
            result.generations[0].message.response_metadata = {
                **result.llm_output,
                **result.generations[0].message.response_metadata,
            }
        if check_cache and llm_cache:
            llm_cache.update(prompt, llm_string, result.generations)
        return result

    async def _agenerate_with_cache(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if isinstance(self.cache, BaseCache):
            llm_cache = self.cache
        else:
            llm_cache = get_llm_cache()
        # We should check the cache unless it's explicitly set to False
        # A None cache means we should use the default global cache
        # if it's configured.
        check_cache = self.cache or self.cache is None
        if check_cache:
            if llm_cache:
                llm_string = self._get_llm_string(stop=stop, **kwargs)
                prompt = dumps(messages)
                cache_val = await llm_cache.alookup(prompt, llm_string)
                if isinstance(cache_val, list):
                    return ChatResult(generations=cache_val)
            elif self.cache is None:
                pass
            else:
                raise ValueError(
                    "Asked to cache, but no cache found at `langchain.cache`."
                )

        # Apply the rate limiter after checking the cache, since
        # we usually don't want to rate limit cache lookups, but
        # we do want to rate limit API requests.
        if self.rate_limiter:
            self.rate_limiter.acquire(blocking=True)

        # If stream is not explicitly set, check if implicitly requested by
        # astream_events() or astream_log(). Bail out if _astream not implemented
        if (
            type(self)._astream != BaseChatModel._astream
            or type(self)._stream != BaseChatModel._stream
        ) and kwargs.pop(
            "stream",
            (
                next(
                    (
                        True
                        for h in run_manager.handlers
                        if isinstance(h, _StreamingCallbackHandler)
                    ),
                    False,
                )
                if run_manager
                else False
            ),
        ):
            chunks: List[ChatGenerationChunk] = []
            async for chunk in self._astream(messages, stop=stop, **kwargs):
                chunk.message.response_metadata = _gen_info_and_msg_metadata(chunk)
                if run_manager:
                    if chunk.message.id is None:
                        chunk.message.id = f"run-{run_manager.run_id}"
                    await run_manager.on_llm_new_token(
                        cast(str, chunk.message.content), chunk=chunk
                    )
                chunks.append(chunk)
            result = generate_from_stream(iter(chunks))
        else:
            if inspect.signature(self._agenerate).parameters.get("run_manager"):
                result = await self._agenerate(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
            else:
                result = await self._agenerate(messages, stop=stop, **kwargs)

        # Add response metadata to each generation
        for idx, generation in enumerate(result.generations):
            if run_manager and generation.message.id is None:
                generation.message.id = f"run-{run_manager.run_id}-{idx}"
            generation.message.response_metadata = _gen_info_and_msg_metadata(
                generation
            )
        if len(result.generations) == 1 and result.llm_output is not None:
            result.generations[0].message.response_metadata = {
                **result.llm_output,
                **result.generations[0].message.response_metadata,
            }
        if check_cache and llm_cache:
            await llm_cache.aupdate(prompt, llm_string, result.generations)
        return result

    @abstractmethod
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call"""

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call"""
        return await run_in_executor(
            None,
            self._generate,
            messages,
            stop,
            run_manager.get_sync() if run_manager else None,
            **kwargs,
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        raise NotImplementedError()

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        iterator = await run_in_executor(
            None,
            self._stream,
            messages,
            stop,
            run_manager.get_sync() if run_manager else None,
            **kwargs,
        )
        done = object()
        while True:
            item = await run_in_executor(
                None,
                next,
                iterator,
                done,  # type: ignore[call-arg, arg-type]
            )
            if item is done:
                break
            yield item  # type: ignore[misc]

    @deprecated("0.1.7", alternative="invoke", removal="0.3.0")
    def __call__(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> BaseMessage:
        generation = self.generate(
            [messages], stop=stop, callbacks=callbacks, **kwargs
        ).generations[0][0]
        if isinstance(generation, ChatGeneration):
            return generation.message
        else:
            raise ValueError("Unexpected generation type")

    async def _call_async(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> BaseMessage:
        result = await self.agenerate(
            [messages], stop=stop, callbacks=callbacks, **kwargs
        )
        generation = result.generations[0][0]
        if isinstance(generation, ChatGeneration):
            return generation.message
        else:
            raise ValueError("Unexpected generation type")

    @deprecated("0.1.7", alternative="invoke", removal="0.3.0")
    def call_as_llm(
        self, message: str, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> str:
        return self.predict(message, stop=stop, **kwargs)

    @deprecated("0.1.7", alternative="invoke", removal="0.3.0")
    def predict(
        self, text: str, *, stop: Optional[Sequence[str]] = None, **kwargs: Any
    ) -> str:
        if stop is None:
            _stop = None
        else:
            _stop = list(stop)
        result = self([HumanMessage(content=text)], stop=_stop, **kwargs)
        if isinstance(result.content, str):
            return result.content
        else:
            raise ValueError("Cannot use predict when output is not a string.")

    @deprecated("0.1.7", alternative="invoke", removal="0.3.0")
    def predict_messages(
        self,
        messages: List[BaseMessage],
        *,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        if stop is None:
            _stop = None
        else:
            _stop = list(stop)
        return self(messages, stop=_stop, **kwargs)

    @deprecated("0.1.7", alternative="ainvoke", removal="0.3.0")
    async def apredict(
        self, text: str, *, stop: Optional[Sequence[str]] = None, **kwargs: Any
    ) -> str:
        if stop is None:
            _stop = None
        else:
            _stop = list(stop)
        result = await self._call_async(
            [HumanMessage(content=text)], stop=_stop, **kwargs
        )
        if isinstance(result.content, str):
            return result.content
        else:
            raise ValueError("Cannot use predict when output is not a string.")

    @deprecated("0.1.7", alternative="ainvoke", removal="0.3.0")
    async def apredict_messages(
        self,
        messages: List[BaseMessage],
        *,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        if stop is None:
            _stop = None
        else:
            _stop = list(stop)
        return await self._call_async(messages, stop=_stop, **kwargs)

    @property
    @abstractmethod
    def _llm_type(self) -> str:
        """Return type of chat model."""

    def dict(self, **kwargs: Any) -> Dict:
        """Return a dictionary of the LLM."""
        starter_dict = dict(self._identifying_params)
        starter_dict["_type"] = self._llm_type
        return starter_dict

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        raise NotImplementedError()

    def with_structured_output(
        self,
        schema: Union[Dict, Type],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema:
                The output schema. Can be passed in as:
                    - an OpenAI function/tool schema,
                    - a JSON Schema,
                    - a TypedDict class (support added in 0.2.26),
                    - or a Pydantic class.
                If ``schema`` is a Pydantic class then the model output will be a
                Pydantic instance of that class, and the model-generated fields will be
                validated by the Pydantic class. Otherwise the model output will be a
                dict and will not be validated. See :meth:`langchain_core.utils.function_calling.convert_to_openai_tool`
                for more on how to properly specify types and descriptions of
                schema fields when specifying a Pydantic or TypedDict class.

                .. versionchanged:: 0.2.26

                        Added support for TypedDict class.

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

                from langchain_core.pydantic_v1 import BaseModel

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

                from langchain_core.pydantic_v1 import BaseModel

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

                from langchain_core.pydantic_v1 import BaseModel
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
        """  # noqa: E501
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")

        from langchain_core.output_parsers.openai_tools import (
            JsonOutputKeyToolsParser,
            PydanticToolsParser,
        )

        if self.bind_tools is BaseChatModel.bind_tools:
            raise NotImplementedError(
                "with_structured_output is not implemented for this model."
            )
        llm = self.bind_tools([schema], tool_choice="any")
        if isinstance(schema, type) and is_basemodel_subclass(schema):
            output_parser: OutputParserLike = PydanticToolsParser(
                tools=[cast(TypeBaseModel, schema)], first_tool_only=True
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


class SimpleChatModel(BaseChatModel):
    """Simplified implementation for a chat model to inherit from.

    **Note** This implementation is primarily here for backwards compatibility.
        For new implementations, please use `BaseChatModel` directly.
    """

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        output_str = self._call(messages, stop=stop, run_manager=run_manager, **kwargs)
        message = AIMessage(content=output_str)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @abstractmethod
    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Simpler interface."""

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return await run_in_executor(
            None,
            self._generate,
            messages,
            stop=stop,
            run_manager=run_manager.get_sync() if run_manager else None,
            **kwargs,
        )


def _gen_info_and_msg_metadata(
    generation: Union[ChatGeneration, ChatGenerationChunk],
) -> dict:
    return {
        **(generation.generation_info or {}),
        **generation.message.response_metadata,
    }


def _cleanup_llm_representation(serialized: Any, depth: int) -> None:
    """Remove non-serializable objects from a serialized object."""
    if depth > 100:  # Don't cooperate for pathological cases
        return

    if not isinstance(serialized, dict):
        return

    if "type" in serialized and serialized["type"] == "not_implemented":
        if "repr" in serialized:
            del serialized["repr"]

    if "graph" in serialized:
        del serialized["graph"]

    if "kwargs" in serialized:
        kwargs = serialized["kwargs"]

        for value in kwargs.values():
            _cleanup_llm_representation(value, depth + 1)
