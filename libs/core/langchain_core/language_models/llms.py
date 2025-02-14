"""Base interface for large language models to expose."""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import uuid
import warnings
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path
from typing import (
    Any,
    Callable,
    Optional,
    Union,
    cast,
)

import yaml
from pydantic import ConfigDict, Field, model_validator
from tenacity import (
    RetryCallState,
    before_sleep_log,
    retry,
    retry_base,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from typing_extensions import override

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
from langchain_core.language_models.base import (
    BaseLanguageModel,
    LangSmithParams,
    LanguageModelInput,
)
from langchain_core.load import dumpd
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    convert_to_messages,
    get_buffer_string,
)
from langchain_core.outputs import Generation, GenerationChunk, LLMResult, RunInfo
from langchain_core.prompt_values import ChatPromptValue, PromptValue, StringPromptValue
from langchain_core.runnables import RunnableConfig, ensure_config, get_config_list
from langchain_core.runnables.config import run_in_executor

logger = logging.getLogger(__name__)


@functools.lru_cache
def _log_error_once(msg: str) -> None:
    """Log an error once."""
    logger.error(msg)


def create_base_retry_decorator(
    error_types: list[type[BaseException]],
    max_retries: int = 1,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:
    """Create a retry decorator for a given LLM and provided
     a list of error types.

    Args:
        error_types: List of error types to retry on.
        max_retries: Number of retries. Default is 1.
        run_manager: Callback manager for the run. Default is None.

    Returns:
        A retry decorator.

    Raises:
        ValueError: If the cache is not set and cache is True.
    """
    _logging = before_sleep_log(logger, logging.WARNING)

    def _before_sleep(retry_state: RetryCallState) -> None:
        _logging(retry_state)
        if run_manager:
            if isinstance(run_manager, AsyncCallbackManagerForLLMRun):
                coro = run_manager.on_retry(retry_state)
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(coro)
                    else:
                        asyncio.run(coro)
                except Exception as e:
                    _log_error_once(f"Error in on_retry: {e}")
            else:
                run_manager.on_retry(retry_state)

    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    retry_instance: retry_base = retry_if_exception_type(error_types[0])
    for error in error_types[1:]:
        retry_instance = retry_instance | retry_if_exception_type(error)
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=retry_instance,
        before_sleep=_before_sleep,
    )


def _resolve_cache(cache: Union[BaseCache, bool, None]) -> Optional[BaseCache]:
    """Resolve the cache."""
    if isinstance(cache, BaseCache):
        llm_cache = cache
    elif cache is None:
        llm_cache = get_llm_cache()
    elif cache is True:
        llm_cache = get_llm_cache()
        if llm_cache is None:
            msg = (
                "No global cache was configured. Use `set_llm_cache`."
                "to set a global cache if you want to use a global cache."
                "Otherwise either pass a cache object or set cache to False/None"
            )
            raise ValueError(msg)
    elif cache is False:
        llm_cache = None
    else:
        msg = f"Unsupported cache value {cache}"
        raise ValueError(msg)
    return llm_cache


def get_prompts(
    params: dict[str, Any],
    prompts: list[str],
    cache: Optional[Union[BaseCache, bool, None]] = None,
) -> tuple[dict[int, list], str, list[int], list[str]]:
    """Get prompts that are already cached.

    Args:
        params: Dictionary of parameters.
        prompts: List of prompts.
        cache: Cache object. Default is None.

    Returns:
        A tuple of existing prompts, llm_string, missing prompt indexes,
            and missing prompts.

    Raises:
        ValueError: If the cache is not set and cache is True.
    """
    llm_string = str(sorted(params.items()))
    missing_prompts = []
    missing_prompt_idxs = []
    existing_prompts = {}

    llm_cache = _resolve_cache(cache)
    for i, prompt in enumerate(prompts):
        if llm_cache:
            cache_val = llm_cache.lookup(prompt, llm_string)
            if isinstance(cache_val, list):
                existing_prompts[i] = cache_val
            else:
                missing_prompts.append(prompt)
                missing_prompt_idxs.append(i)
    return existing_prompts, llm_string, missing_prompt_idxs, missing_prompts


async def aget_prompts(
    params: dict[str, Any],
    prompts: list[str],
    cache: Optional[Union[BaseCache, bool, None]] = None,
) -> tuple[dict[int, list], str, list[int], list[str]]:
    """Get prompts that are already cached. Async version.

    Args:
        params: Dictionary of parameters.
        prompts: List of prompts.
        cache: Cache object. Default is None.

    Returns:
        A tuple of existing prompts, llm_string, missing prompt indexes,
            and missing prompts.

    Raises:
        ValueError: If the cache is not set and cache is True.
    """
    llm_string = str(sorted(params.items()))
    missing_prompts = []
    missing_prompt_idxs = []
    existing_prompts = {}
    llm_cache = _resolve_cache(cache)
    for i, prompt in enumerate(prompts):
        if llm_cache:
            cache_val = await llm_cache.alookup(prompt, llm_string)
            if isinstance(cache_val, list):
                existing_prompts[i] = cache_val
            else:
                missing_prompts.append(prompt)
                missing_prompt_idxs.append(i)
    return existing_prompts, llm_string, missing_prompt_idxs, missing_prompts


def update_cache(
    cache: Union[BaseCache, bool, None],
    existing_prompts: dict[int, list],
    llm_string: str,
    missing_prompt_idxs: list[int],
    new_results: LLMResult,
    prompts: list[str],
) -> Optional[dict]:
    """Update the cache and get the LLM output.

    Args:
        cache: Cache object.
        existing_prompts: Dictionary of existing prompts.
        llm_string: LLM string.
        missing_prompt_idxs: List of missing prompt indexes.
        new_results: LLMResult object.
        prompts: List of prompts.

    Returns:
        LLM output.

    Raises:
        ValueError: If the cache is not set and cache is True.
    """
    llm_cache = _resolve_cache(cache)
    for i, result in enumerate(new_results.generations):
        existing_prompts[missing_prompt_idxs[i]] = result
        prompt = prompts[missing_prompt_idxs[i]]
        if llm_cache is not None:
            llm_cache.update(prompt, llm_string, result)
    llm_output = new_results.llm_output
    return llm_output


async def aupdate_cache(
    cache: Union[BaseCache, bool, None],
    existing_prompts: dict[int, list],
    llm_string: str,
    missing_prompt_idxs: list[int],
    new_results: LLMResult,
    prompts: list[str],
) -> Optional[dict]:
    """Update the cache and get the LLM output. Async version.

    Args:
        cache: Cache object.
        existing_prompts: Dictionary of existing prompts.
        llm_string: LLM string.
        missing_prompt_idxs: List of missing prompt indexes.
        new_results: LLMResult object.
        prompts: List of prompts.

    Returns:
        LLM output.

    Raises:
        ValueError: If the cache is not set and cache is True.
    """
    llm_cache = _resolve_cache(cache)
    for i, result in enumerate(new_results.generations):
        existing_prompts[missing_prompt_idxs[i]] = result
        prompt = prompts[missing_prompt_idxs[i]]
        if llm_cache:
            await llm_cache.aupdate(prompt, llm_string, result)
    llm_output = new_results.llm_output
    return llm_output


class BaseLLM(BaseLanguageModel[str], ABC):
    """Base LLM abstract interface.

    It should take in a prompt and return a string.
    """

    callback_manager: Optional[BaseCallbackManager] = Field(default=None, exclude=True)
    """[DEPRECATED]"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="before")
    @classmethod
    def raise_deprecation(cls, values: dict) -> Any:
        """Raise deprecation warning if callback_manager is used."""
        if values.get("callback_manager") is not None:
            warnings.warn(
                "callback_manager is deprecated. Please use callbacks instead.",
                DeprecationWarning,
                stacklevel=5,
            )
            values["callbacks"] = values.pop("callback_manager", None)
        return values

    @functools.cached_property
    def _serialized(self) -> dict[str, Any]:
        return dumpd(self)

    # --- Runnable methods ---

    @property
    @override
    def OutputType(self) -> type[str]:
        """Get the input type for this runnable."""
        return str

    def _convert_input(self, input: LanguageModelInput) -> PromptValue:
        if isinstance(input, PromptValue):
            return input
        elif isinstance(input, str):
            return StringPromptValue(text=input)
        elif isinstance(input, Sequence):
            return ChatPromptValue(messages=convert_to_messages(input))
        else:
            msg = (
                f"Invalid input type {type(input)}. "
                "Must be a PromptValue, str, or list of BaseMessages."
            )
            raise ValueError(msg)  # noqa: TRY004

    def _get_ls_params(
        self,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        # get default provider from class name
        default_provider = self.__class__.__name__
        default_provider = default_provider.removesuffix("LLM")
        default_provider = default_provider.lower()

        ls_params = LangSmithParams(ls_provider=default_provider, ls_model_type="llm")
        if stop:
            ls_params["ls_stop"] = stop

        # model
        if hasattr(self, "model") and isinstance(self.model, str):
            ls_params["ls_model_name"] = self.model
        elif hasattr(self, "model_name") and isinstance(self.model_name, str):
            ls_params["ls_model_name"] = self.model_name

        # temperature
        if "temperature" in kwargs and isinstance(kwargs["temperature"], float):
            ls_params["ls_temperature"] = kwargs["temperature"]
        elif hasattr(self, "temperature") and isinstance(self.temperature, float):
            ls_params["ls_temperature"] = self.temperature

        # max_tokens
        if "max_tokens" in kwargs and isinstance(kwargs["max_tokens"], int):
            ls_params["ls_max_tokens"] = kwargs["max_tokens"]
        elif hasattr(self, "max_tokens") and isinstance(self.max_tokens, int):
            ls_params["ls_max_tokens"] = self.max_tokens

        return ls_params

    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> str:
        config = ensure_config(config)
        return (
            self.generate_prompt(
                [self._convert_input(input)],
                stop=stop,
                callbacks=config.get("callbacks"),
                tags=config.get("tags"),
                metadata=config.get("metadata"),
                run_name=config.get("run_name"),
                run_id=config.pop("run_id", None),
                **kwargs,
            )
            .generations[0][0]
            .text
        )

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> str:
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
        return llm_result.generations[0][0].text

    def batch(
        self,
        inputs: list[LanguageModelInput],
        config: Optional[Union[RunnableConfig, list[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[str]:
        if not inputs:
            return []

        config = get_config_list(config, len(inputs))
        max_concurrency = config[0].get("max_concurrency")

        if max_concurrency is None:
            try:
                llm_result = self.generate_prompt(
                    [self._convert_input(input) for input in inputs],
                    callbacks=[c.get("callbacks") for c in config],
                    tags=[c.get("tags") for c in config],
                    metadata=[c.get("metadata") for c in config],
                    run_name=[c.get("run_name") for c in config],
                    **kwargs,
                )
                return [g[0].text for g in llm_result.generations]
            except Exception as e:
                if return_exceptions:
                    return cast(list[str], [e for _ in inputs])
                else:
                    raise
        else:
            batches = [
                inputs[i : i + max_concurrency]
                for i in range(0, len(inputs), max_concurrency)
            ]
            config = [{**c, "max_concurrency": None} for c in config]  # type: ignore[misc]
            return [
                output
                for i, batch in enumerate(batches)
                for output in self.batch(
                    batch,
                    config=config[i * max_concurrency : (i + 1) * max_concurrency],
                    return_exceptions=return_exceptions,
                    **kwargs,
                )
            ]

    async def abatch(
        self,
        inputs: list[LanguageModelInput],
        config: Optional[Union[RunnableConfig, list[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[str]:
        if not inputs:
            return []
        config = get_config_list(config, len(inputs))
        max_concurrency = config[0].get("max_concurrency")

        if max_concurrency is None:
            try:
                llm_result = await self.agenerate_prompt(
                    [self._convert_input(input) for input in inputs],
                    callbacks=[c.get("callbacks") for c in config],
                    tags=[c.get("tags") for c in config],
                    metadata=[c.get("metadata") for c in config],
                    run_name=[c.get("run_name") for c in config],
                    **kwargs,
                )
                return [g[0].text for g in llm_result.generations]
            except Exception as e:
                if return_exceptions:
                    return cast(list[str], [e for _ in inputs])
                else:
                    raise
        else:
            batches = [
                inputs[i : i + max_concurrency]
                for i in range(0, len(inputs), max_concurrency)
            ]
            config = [{**c, "max_concurrency": None} for c in config]  # type: ignore[misc]
            return [
                output
                for i, batch in enumerate(batches)
                for output in await self.abatch(
                    batch,
                    config=config[i * max_concurrency : (i + 1) * max_concurrency],
                    return_exceptions=return_exceptions,
                    **kwargs,
                )
            ]

    def stream(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        if type(self)._stream == BaseLLM._stream:
            # model doesn't implement streaming, so use default implementation
            yield self.invoke(input, config=config, stop=stop, **kwargs)
        else:
            prompt = self._convert_input(input).to_string()
            config = ensure_config(config)
            params = self.dict()
            params["stop"] = stop
            params = {**params, **kwargs}
            options = {"stop": stop}
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
            (run_manager,) = callback_manager.on_llm_start(
                self._serialized,
                [prompt],
                invocation_params=params,
                options=options,
                name=config.get("run_name"),
                run_id=config.pop("run_id", None),
                batch_size=1,
            )
            generation: Optional[GenerationChunk] = None
            try:
                for chunk in self._stream(
                    prompt, stop=stop, run_manager=run_manager, **kwargs
                ):
                    yield chunk.text
                    if generation is None:
                        generation = chunk
                    else:
                        generation += chunk
            except BaseException as e:
                run_manager.on_llm_error(
                    e,
                    response=LLMResult(
                        generations=[[generation]] if generation else []
                    ),
                )
                raise

            if generation is None:
                err = ValueError("No generation chunks were returned")
                run_manager.on_llm_error(err, response=LLMResult(generations=[]))
                raise err

            run_manager.on_llm_end(LLMResult(generations=[[generation]]))

    async def astream(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        if (
            type(self)._astream is BaseLLM._astream
            and type(self)._stream is BaseLLM._stream
        ):
            yield await self.ainvoke(input, config=config, stop=stop, **kwargs)
            return

        prompt = self._convert_input(input).to_string()
        config = ensure_config(config)
        params = self.dict()
        params["stop"] = stop
        params = {**params, **kwargs}
        options = {"stop": stop}
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
        (run_manager,) = await callback_manager.on_llm_start(
            self._serialized,
            [prompt],
            invocation_params=params,
            options=options,
            name=config.get("run_name"),
            run_id=config.pop("run_id", None),
            batch_size=1,
        )
        generation: Optional[GenerationChunk] = None
        try:
            async for chunk in self._astream(
                prompt,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            ):
                yield chunk.text
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
        except BaseException as e:
            await run_manager.on_llm_error(
                e,
                response=LLMResult(generations=[[generation]] if generation else []),
            )
            raise

        if generation is None:
            err = ValueError("No generation chunks were returned")
            await run_manager.on_llm_error(err, response=LLMResult(generations=[]))
            raise err

        await run_manager.on_llm_end(LLMResult(generations=[[generation]]))

    # --- Custom methods ---

    @abstractmethod
    def _generate(
        self,
        prompts: list[str],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompts."""

    async def _agenerate(
        self,
        prompts: list[str],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompts."""
        return await run_in_executor(
            None,
            self._generate,
            prompts,
            stop,
            run_manager.get_sync() if run_manager else None,
            **kwargs,
        )

    def _stream(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream the LLM on the given prompt.

        This method should be overridden by subclasses that support streaming.

        If not implemented, the default behavior of calls to stream will be to
        fallback to the non-streaming version of the model and return
        the output as a single chunk.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            An iterator of GenerationChunks.
        """
        raise NotImplementedError

    async def _astream(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        """An async version of the _stream method.

        The default implementation uses the synchronous _stream method and wraps it in
        an async iterator. Subclasses that need to provide a true async implementation
        should override this method.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            An async iterator of GenerationChunks.
        """
        iterator = await run_in_executor(
            None,
            self._stream,
            prompt,
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

    def generate_prompt(
        self,
        prompts: list[PromptValue],
        stop: Optional[list[str]] = None,
        callbacks: Optional[Union[Callbacks, list[Callbacks]]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        prompt_strings = [p.to_string() for p in prompts]
        return self.generate(prompt_strings, stop=stop, callbacks=callbacks, **kwargs)

    async def agenerate_prompt(
        self,
        prompts: list[PromptValue],
        stop: Optional[list[str]] = None,
        callbacks: Optional[Union[Callbacks, list[Callbacks]]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        prompt_strings = [p.to_string() for p in prompts]
        return await self.agenerate(
            prompt_strings, stop=stop, callbacks=callbacks, **kwargs
        )

    def _generate_helper(
        self,
        prompts: list[str],
        stop: Optional[list[str]],
        run_managers: list[CallbackManagerForLLMRun],
        new_arg_supported: bool,
        **kwargs: Any,
    ) -> LLMResult:
        try:
            output = (
                self._generate(
                    prompts,
                    stop=stop,
                    # TODO: support multiple run managers
                    run_manager=run_managers[0] if run_managers else None,
                    **kwargs,
                )
                if new_arg_supported
                else self._generate(prompts, stop=stop)
            )
        except BaseException as e:
            for run_manager in run_managers:
                run_manager.on_llm_error(e, response=LLMResult(generations=[]))
            raise
        flattened_outputs = output.flatten()
        for manager, flattened_output in zip(run_managers, flattened_outputs):
            manager.on_llm_end(flattened_output)
        if run_managers:
            output.run = [
                RunInfo(run_id=run_manager.run_id) for run_manager in run_managers
            ]
        return output

    def generate(
        self,
        prompts: list[str],
        stop: Optional[list[str]] = None,
        callbacks: Optional[Union[Callbacks, list[Callbacks]]] = None,
        *,
        tags: Optional[Union[list[str], list[list[str]]]] = None,
        metadata: Optional[Union[dict[str, Any], list[dict[str, Any]]]] = None,
        run_name: Optional[Union[str, list[str]]] = None,
        run_id: Optional[Union[uuid.UUID, list[Optional[uuid.UUID]]]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Pass a sequence of prompts to a model and return generations.

        This method should make use of batched calls for models that expose a batched
        API.

        Use this method when you want to:
            1. take advantage of batched calls,
            2. need more output from the model than just the top generated value,
            3. are building chains that are agnostic to the underlying language model
                type (e.g., pure text completion models vs chat models).

        Args:
            prompts: List of string prompts.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings.
            callbacks: Callbacks to pass through. Used for executing additional
                functionality, such as logging or streaming, throughout generation.
            tags: List of tags to associate with each prompt. If provided, the length
                of the list must match the length of the prompts list.
            metadata: List of metadata dictionaries to associate with each prompt. If
                provided, the length of the list must match the length of the prompts
                list.
            run_name: List of run names to associate with each prompt. If provided, the
                length of the list must match the length of the prompts list.
            run_id: List of run IDs to associate with each prompt. If provided, the
                length of the list must match the length of the prompts list.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            An LLMResult, which contains a list of candidate Generations for each input
                prompt and additional model provider-specific output.
        """
        if not isinstance(prompts, list):
            msg = (
                "Argument 'prompts' is expected to be of type List[str], received"
                f" argument of type {type(prompts)}."
            )
            raise ValueError(msg)  # noqa: TRY004
        # Create callback managers
        if isinstance(metadata, list):
            metadata = [
                {
                    **(meta or {}),
                    **self._get_ls_params(stop=stop, **kwargs),
                }
                for meta in metadata
            ]
        elif isinstance(metadata, dict):
            metadata = {
                **(metadata or {}),
                **self._get_ls_params(stop=stop, **kwargs),
            }
        else:
            pass
        if (
            isinstance(callbacks, list)
            and callbacks
            and (
                isinstance(callbacks[0], (list, BaseCallbackManager))
                or callbacks[0] is None
            )
        ):
            # We've received a list of callbacks args to apply to each input
            if len(callbacks) != len(prompts):
                msg = "callbacks must be the same length as prompts"
                raise ValueError(msg)
            if tags is not None and not (
                isinstance(tags, list) and len(tags) == len(prompts)
            ):
                msg = "tags must be a list of the same length as prompts"
                raise ValueError(msg)
            if metadata is not None and not (
                isinstance(metadata, list) and len(metadata) == len(prompts)
            ):
                msg = "metadata must be a list of the same length as prompts"
                raise ValueError(msg)
            if run_name is not None and not (
                isinstance(run_name, list) and len(run_name) == len(prompts)
            ):
                msg = "run_name must be a list of the same length as prompts"
                raise ValueError(msg)
            callbacks = cast(list[Callbacks], callbacks)
            tags_list = cast(list[Optional[list[str]]], tags or ([None] * len(prompts)))
            metadata_list = cast(
                list[Optional[dict[str, Any]]], metadata or ([{}] * len(prompts))
            )
            run_name_list = run_name or cast(
                list[Optional[str]], ([None] * len(prompts))
            )
            callback_managers = [
                CallbackManager.configure(
                    callback,
                    self.callbacks,
                    self.verbose,
                    tag,
                    self.tags,
                    meta,
                    self.metadata,
                )
                for callback, tag, meta in zip(callbacks, tags_list, metadata_list)
            ]
        else:
            # We've received a single callbacks arg to apply to all inputs
            callback_managers = [
                CallbackManager.configure(
                    cast(Callbacks, callbacks),
                    self.callbacks,
                    self.verbose,
                    cast(list[str], tags),
                    self.tags,
                    cast(dict[str, Any], metadata),
                    self.metadata,
                )
            ] * len(prompts)
            run_name_list = [cast(Optional[str], run_name)] * len(prompts)
        run_ids_list = self._get_run_ids_list(run_id, prompts)
        params = self.dict()
        params["stop"] = stop
        options = {"stop": stop}
        (
            existing_prompts,
            llm_string,
            missing_prompt_idxs,
            missing_prompts,
        ) = get_prompts(params, prompts, self.cache)
        new_arg_supported = inspect.signature(self._generate).parameters.get(
            "run_manager"
        )
        if (self.cache is None and get_llm_cache() is None) or self.cache is False:
            run_managers = [
                callback_manager.on_llm_start(
                    self._serialized,
                    [prompt],
                    invocation_params=params,
                    options=options,
                    name=run_name,
                    batch_size=len(prompts),
                    run_id=run_id_,
                )[0]
                for callback_manager, prompt, run_name, run_id_ in zip(
                    callback_managers, prompts, run_name_list, run_ids_list
                )
            ]
            output = self._generate_helper(
                prompts, stop, run_managers, bool(new_arg_supported), **kwargs
            )
            return output
        if len(missing_prompts) > 0:
            run_managers = [
                callback_managers[idx].on_llm_start(
                    self._serialized,
                    [prompts[idx]],
                    invocation_params=params,
                    options=options,
                    name=run_name_list[idx],
                    batch_size=len(missing_prompts),
                )[0]
                for idx in missing_prompt_idxs
            ]
            new_results = self._generate_helper(
                missing_prompts, stop, run_managers, bool(new_arg_supported), **kwargs
            )
            llm_output = update_cache(
                self.cache,
                existing_prompts,
                llm_string,
                missing_prompt_idxs,
                new_results,
                prompts,
            )
            run_info = (
                [RunInfo(run_id=run_manager.run_id) for run_manager in run_managers]
                if run_managers
                else None
            )
        else:
            llm_output = {}
            run_info = None
        generations = [existing_prompts[i] for i in range(len(prompts))]
        return LLMResult(generations=generations, llm_output=llm_output, run=run_info)

    @staticmethod
    def _get_run_ids_list(
        run_id: Optional[Union[uuid.UUID, list[Optional[uuid.UUID]]]], prompts: list
    ) -> list:
        if run_id is None:
            return [None] * len(prompts)
        if isinstance(run_id, list):
            if len(run_id) != len(prompts):
                msg = (
                    "Number of manually provided run_id's does not match batch length."
                    f" {len(run_id)} != {len(prompts)}"
                )
                raise ValueError(msg)
            return run_id
        return [run_id] + [None] * (len(prompts) - 1)

    async def _agenerate_helper(
        self,
        prompts: list[str],
        stop: Optional[list[str]],
        run_managers: list[AsyncCallbackManagerForLLMRun],
        new_arg_supported: bool,
        **kwargs: Any,
    ) -> LLMResult:
        try:
            output = (
                await self._agenerate(
                    prompts,
                    stop=stop,
                    run_manager=run_managers[0] if run_managers else None,
                    **kwargs,
                )
                if new_arg_supported
                else await self._agenerate(prompts, stop=stop)
            )
        except BaseException as e:
            await asyncio.gather(
                *[
                    run_manager.on_llm_error(e, response=LLMResult(generations=[]))
                    for run_manager in run_managers
                ]
            )
            raise
        flattened_outputs = output.flatten()
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

    async def agenerate(
        self,
        prompts: list[str],
        stop: Optional[list[str]] = None,
        callbacks: Optional[Union[Callbacks, list[Callbacks]]] = None,
        *,
        tags: Optional[Union[list[str], list[list[str]]]] = None,
        metadata: Optional[Union[dict[str, Any], list[dict[str, Any]]]] = None,
        run_name: Optional[Union[str, list[str]]] = None,
        run_id: Optional[Union[uuid.UUID, list[Optional[uuid.UUID]]]] = None,
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
            prompts: List of string prompts.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings.
            callbacks: Callbacks to pass through. Used for executing additional
                functionality, such as logging or streaming, throughout generation.
            tags: List of tags to associate with each prompt. If provided, the length
                of the list must match the length of the prompts list.
            metadata: List of metadata dictionaries to associate with each prompt. If
                provided, the length of the list must match the length of the prompts
                list.
            run_name: List of run names to associate with each prompt. If provided, the
                length of the list must match the length of the prompts list.
            run_id: List of run IDs to associate with each prompt. If provided, the
                length of the list must match the length of the prompts list.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            An LLMResult, which contains a list of candidate Generations for each input
                prompt and additional model provider-specific output.
        """
        if isinstance(metadata, list):
            metadata = [
                {
                    **(meta or {}),
                    **self._get_ls_params(stop=stop, **kwargs),
                }
                for meta in metadata
            ]
        elif isinstance(metadata, dict):
            metadata = {
                **(metadata or {}),
                **self._get_ls_params(stop=stop, **kwargs),
            }
        else:
            pass
        # Create callback managers
        if isinstance(callbacks, list) and (
            isinstance(callbacks[0], (list, BaseCallbackManager))
            or callbacks[0] is None
        ):
            # We've received a list of callbacks args to apply to each input
            if len(callbacks) != len(prompts):
                msg = "callbacks must be the same length as prompts"
                raise ValueError(msg)
            if tags is not None and not (
                isinstance(tags, list) and len(tags) == len(prompts)
            ):
                msg = "tags must be a list of the same length as prompts"
                raise ValueError(msg)
            if metadata is not None and not (
                isinstance(metadata, list) and len(metadata) == len(prompts)
            ):
                msg = "metadata must be a list of the same length as prompts"
                raise ValueError(msg)
            if run_name is not None and not (
                isinstance(run_name, list) and len(run_name) == len(prompts)
            ):
                msg = "run_name must be a list of the same length as prompts"
                raise ValueError(msg)
            callbacks = cast(list[Callbacks], callbacks)
            tags_list = cast(list[Optional[list[str]]], tags or ([None] * len(prompts)))
            metadata_list = cast(
                list[Optional[dict[str, Any]]], metadata or ([{}] * len(prompts))
            )
            run_name_list = run_name or cast(
                list[Optional[str]], ([None] * len(prompts))
            )
            callback_managers = [
                AsyncCallbackManager.configure(
                    callback,
                    self.callbacks,
                    self.verbose,
                    tag,
                    self.tags,
                    meta,
                    self.metadata,
                )
                for callback, tag, meta in zip(callbacks, tags_list, metadata_list)
            ]
        else:
            # We've received a single callbacks arg to apply to all inputs
            callback_managers = [
                AsyncCallbackManager.configure(
                    cast(Callbacks, callbacks),
                    self.callbacks,
                    self.verbose,
                    cast(list[str], tags),
                    self.tags,
                    cast(dict[str, Any], metadata),
                    self.metadata,
                )
            ] * len(prompts)
            run_name_list = [cast(Optional[str], run_name)] * len(prompts)
        run_ids_list = self._get_run_ids_list(run_id, prompts)
        params = self.dict()
        params["stop"] = stop
        options = {"stop": stop}
        (
            existing_prompts,
            llm_string,
            missing_prompt_idxs,
            missing_prompts,
        ) = await aget_prompts(params, prompts, self.cache)

        # Verify whether the cache is set, and if the cache is set,
        # verify whether the cache is available.
        new_arg_supported = inspect.signature(self._agenerate).parameters.get(
            "run_manager"
        )
        if (self.cache is None and get_llm_cache() is None) or self.cache is False:
            run_managers = await asyncio.gather(
                *[
                    callback_manager.on_llm_start(
                        self._serialized,
                        [prompt],
                        invocation_params=params,
                        options=options,
                        name=run_name,
                        batch_size=len(prompts),
                        run_id=run_id_,
                    )
                    for callback_manager, prompt, run_name, run_id_ in zip(
                        callback_managers, prompts, run_name_list, run_ids_list
                    )
                ]
            )
            run_managers = [r[0] for r in run_managers]  # type: ignore[misc]
            output = await self._agenerate_helper(
                prompts,
                stop,
                run_managers,  # type: ignore[arg-type]
                bool(new_arg_supported),
                **kwargs,  # type: ignore[arg-type]
            )
            return output
        if len(missing_prompts) > 0:
            run_managers = await asyncio.gather(
                *[
                    callback_managers[idx].on_llm_start(
                        self._serialized,
                        [prompts[idx]],
                        invocation_params=params,
                        options=options,
                        name=run_name_list[idx],
                        batch_size=len(missing_prompts),
                    )
                    for idx in missing_prompt_idxs
                ]
            )
            run_managers = [r[0] for r in run_managers]  # type: ignore[misc]
            new_results = await self._agenerate_helper(
                missing_prompts,
                stop,
                run_managers,  # type: ignore[arg-type]
                bool(new_arg_supported),
                **kwargs,  # type: ignore[arg-type]
            )
            llm_output = await aupdate_cache(
                self.cache,
                existing_prompts,
                llm_string,
                missing_prompt_idxs,
                new_results,
                prompts,
            )
            run_info = (
                [RunInfo(run_id=run_manager.run_id) for run_manager in run_managers]  # type: ignore[attr-defined]
                if run_managers
                else None
            )
        else:
            llm_output = {}
            run_info = None
        generations = [existing_prompts[i] for i in range(len(prompts))]
        return LLMResult(generations=generations, llm_output=llm_output, run=run_info)

    @deprecated("0.1.7", alternative="invoke", removal="1.0")
    def __call__(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        callbacks: Callbacks = None,
        *,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Check Cache and run the LLM on the given prompt and input.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings.
            callbacks: Callbacks to pass through. Used for executing additional
                functionality, such as logging or streaming, throughout generation.
            tags: List of tags to associate with the prompt.
            metadata: Metadata to associate with the prompt.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            The generated text.

        Raises:
            ValueError: If the prompt is not a string.
        """
        if not isinstance(prompt, str):
            msg = (
                "Argument `prompt` is expected to be a string. Instead found "
                f"{type(prompt)}. If you want to run the LLM on multiple prompts, use "
                "`generate` instead."
            )
            raise ValueError(msg)  # noqa: TRY004
        return (
            self.generate(
                [prompt],
                stop=stop,
                callbacks=callbacks,
                tags=tags,
                metadata=metadata,
                **kwargs,
            )
            .generations[0][0]
            .text
        )

    async def _call_async(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        callbacks: Callbacks = None,
        *,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Check Cache and run the LLM on the given prompt and input."""
        result = await self.agenerate(
            [prompt],
            stop=stop,
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )
        return result.generations[0][0].text

    @deprecated("0.1.7", alternative="invoke", removal="1.0")
    def predict(
        self, text: str, *, stop: Optional[Sequence[str]] = None, **kwargs: Any
    ) -> str:
        _stop = None if stop is None else list(stop)
        return self(text, stop=_stop, **kwargs)

    @deprecated("0.1.7", alternative="invoke", removal="1.0")
    def predict_messages(
        self,
        messages: list[BaseMessage],
        *,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        text = get_buffer_string(messages)
        _stop = None if stop is None else list(stop)
        content = self(text, stop=_stop, **kwargs)
        return AIMessage(content=content)

    @deprecated("0.1.7", alternative="ainvoke", removal="1.0")
    async def apredict(
        self, text: str, *, stop: Optional[Sequence[str]] = None, **kwargs: Any
    ) -> str:
        _stop = None if stop is None else list(stop)
        return await self._call_async(text, stop=_stop, **kwargs)

    @deprecated("0.1.7", alternative="ainvoke", removal="1.0")
    async def apredict_messages(
        self,
        messages: list[BaseMessage],
        *,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        text = get_buffer_string(messages)
        _stop = None if stop is None else list(stop)
        content = await self._call_async(text, stop=_stop, **kwargs)
        return AIMessage(content=content)

    def __str__(self) -> str:
        """Get a string representation of the object for printing."""
        cls_name = f"\033[1m{self.__class__.__name__}\033[0m"
        return f"{cls_name}\nParams: {self._identifying_params}"

    @property
    @abstractmethod
    def _llm_type(self) -> str:
        """Return type of llm."""

    def dict(self, **kwargs: Any) -> dict:
        """Return a dictionary of the LLM."""
        starter_dict = dict(self._identifying_params)
        starter_dict["_type"] = self._llm_type
        return starter_dict

    def save(self, file_path: Union[Path, str]) -> None:
        """Save the LLM.

        Args:
            file_path: Path to file to save the LLM to.

        Raises:
            ValueError: If the file path is not a string or Path object.

        Example:
        .. code-block:: python

            llm.save(file_path="path/llm.yaml")
        """
        # Convert file to Path object.
        save_path = Path(file_path) if isinstance(file_path, str) else file_path

        directory_path = save_path.parent
        directory_path.mkdir(parents=True, exist_ok=True)

        # Fetch dictionary to save
        prompt_dict = self.dict()

        if save_path.suffix == ".json":
            with open(file_path, "w") as f:
                json.dump(prompt_dict, f, indent=4)
        elif save_path.suffix.endswith((".yaml", ".yml")):
            with open(file_path, "w") as f:
                yaml.dump(prompt_dict, f, default_flow_style=False)
        else:
            msg = f"{save_path} must be json or yaml"
            raise ValueError(msg)


class LLM(BaseLLM):
    """Simple interface for implementing a custom LLM.

    You should subclass this class and implement the following:

    - `_call` method: Run the LLM on the given prompt and input (used by `invoke`).
    - `_identifying_params` property: Return a dictionary of the identifying parameters
        This is critical for caching and tracing purposes. Identifying parameters
        is a dict that identifies the LLM.
        It should mostly include a `model_name`.

    Optional: Override the following methods to provide more optimizations:

    - `_acall`: Provide a native async version of the `_call` method.
        If not provided, will delegate to the synchronous version using
        `run_in_executor`. (Used by `ainvoke`).
    - `_stream`: Stream the LLM on the given prompt and input.
        `stream` will use `_stream` if provided, otherwise it
        use `_call` and output will arrive in one chunk.
    - `_astream`: Override to provide a native async version of the `_stream` method.
        `astream` will use `_astream` if provided, otherwise it will implement
        a fallback behavior that will use `_stream` if `_stream` is implemented,
        and use `_acall` if `_stream` is not implemented.

    Please see the following guide for more information on how to
    implement a custom LLM:

    https://python.langchain.com/docs/how_to/custom_llm/
    """

    @abstractmethod
    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input.

        Override this method to implement the LLM logic.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            The model output as a string. SHOULD NOT include the prompt.
        """

    async def _acall(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Async version of the _call method.

        The default implementation delegates to the synchronous _call method using
        `run_in_executor`. Subclasses that need to provide a true async implementation
        should override this method to reduce the overhead of using `run_in_executor`.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            The model output as a string. SHOULD NOT include the prompt.
        """
        return await run_in_executor(
            None,
            self._call,
            prompt,
            stop,
            run_manager.get_sync() if run_manager else None,
            **kwargs,
        )

    def _generate(
        self,
        prompts: list[str],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        # TODO: add caching here.
        generations = []
        new_arg_supported = inspect.signature(self._call).parameters.get("run_manager")
        for prompt in prompts:
            text = (
                self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
                if new_arg_supported
                else self._call(prompt, stop=stop, **kwargs)
            )
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    async def _agenerate(
        self,
        prompts: list[str],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Async run the LLM on the given prompt and input."""
        generations = []
        new_arg_supported = inspect.signature(self._acall).parameters.get("run_manager")
        for prompt in prompts:
            text = (
                await self._acall(prompt, stop=stop, run_manager=run_manager, **kwargs)
                if new_arg_supported
                else await self._acall(prompt, stop=stop, **kwargs)
            )
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)
