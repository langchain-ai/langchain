"""Base interface for large language models to expose."""
from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)

import yaml
from tenacity import (
    RetryCallState,
    before_sleep_log,
    retry,
    retry_base,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from langchain_core._api import deprecated
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
from langchain_core.load import dumpd
from langchain_core.messages import AIMessage, BaseMessage, get_buffer_string
from langchain_core.outputs import Generation, GenerationChunk, LLMResult, RunInfo
from langchain_core.prompt_values import ChatPromptValue, PromptValue, StringPromptValue
from langchain_core.pydantic_v1 import Field, root_validator, validator
from langchain_core.runnables import RunnableConfig, ensure_config, get_config_list
from langchain_core.runnables.config import run_in_executor

logger = logging.getLogger(__name__)


def _get_verbosity() -> bool:
    from langchain_core.globals import get_verbose

    return get_verbose()


@functools.lru_cache
def _log_error_once(msg: str) -> None:
    """Log an error once."""
    logger.error(msg)


def create_base_retry_decorator(
    error_types: List[Type[BaseException]],
    max_retries: int = 1,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:
    """Create a retry decorator for a given LLM and provided list of error types."""

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
        return None

    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    retry_instance: "retry_base" = retry_if_exception_type(error_types[0])
    for error in error_types[1:]:
        retry_instance = retry_instance | retry_if_exception_type(error)
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=retry_instance,
        before_sleep=_before_sleep,
    )


def get_prompts(
    params: Dict[str, Any], prompts: List[str]
) -> Tuple[Dict[int, List], str, List[int], List[str]]:
    """Get prompts that are already cached."""
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    missing_prompts = []
    missing_prompt_idxs = []
    existing_prompts = {}
    llm_cache = get_llm_cache()
    for i, prompt in enumerate(prompts):
        if llm_cache is not None:
            cache_val = llm_cache.lookup(prompt, llm_string)
            if isinstance(cache_val, list):
                existing_prompts[i] = cache_val
            else:
                missing_prompts.append(prompt)
                missing_prompt_idxs.append(i)
    return existing_prompts, llm_string, missing_prompt_idxs, missing_prompts


def update_cache(
    existing_prompts: Dict[int, List],
    llm_string: str,
    missing_prompt_idxs: List[int],
    new_results: LLMResult,
    prompts: List[str],
) -> Optional[dict]:
    """Update the cache and get the LLM output."""
    llm_cache = get_llm_cache()
    for i, result in enumerate(new_results.generations):
        existing_prompts[missing_prompt_idxs[i]] = result
        prompt = prompts[missing_prompt_idxs[i]]
        if llm_cache is not None:
            llm_cache.update(prompt, llm_string, result)
    llm_output = new_results.llm_output
    return llm_output


class BaseLLM(BaseLanguageModel[str], ABC):
    """Base LLM abstract interface.

    It should take in a prompt and return a string."""

    cache: Optional[bool] = None
    """Whether to cache the response."""
    verbose: bool = Field(default_factory=_get_verbosity)
    """Whether to print out response text."""
    callbacks: Callbacks = Field(default=None, exclude=True)
    """Callbacks to add to the run trace."""
    tags: Optional[List[str]] = Field(default=None, exclude=True)
    """Tags to add to the run trace."""
    metadata: Optional[Dict[str, Any]] = Field(default=None, exclude=True)
    """Metadata to add to the run trace."""
    callback_manager: Optional[BaseCallbackManager] = Field(default=None, exclude=True)
    """[DEPRECATED]"""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @root_validator()
    def raise_deprecation(cls, values: Dict) -> Dict:
        """Raise deprecation warning if callback_manager is used."""
        if values.get("callback_manager") is not None:
            warnings.warn(
                "callback_manager is deprecated. Please use callbacks instead.",
                DeprecationWarning,
            )
            values["callbacks"] = values.pop("callback_manager", None)
        return values

    @validator("verbose", pre=True, always=True)
    def set_verbose(cls, verbose: Optional[bool]) -> bool:
        """If verbose is None, set it.

        This allows users to pass in None as verbose to access the global setting.
        """
        if verbose is None:
            return _get_verbosity()
        else:
            return verbose

    # --- Runnable methods ---

    @property
    def OutputType(self) -> Type[str]:
        """Get the input type for this runnable."""
        return str

    def _convert_input(self, input: LanguageModelInput) -> PromptValue:
        if isinstance(input, PromptValue):
            return input
        elif isinstance(input, str):
            return StringPromptValue(text=input)
        elif isinstance(input, Sequence):
            return ChatPromptValue(messages=input)
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
        stop: Optional[List[str]] = None,
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
            **kwargs,
        )
        return llm_result.generations[0][0].text

    def batch(
        self,
        inputs: List[LanguageModelInput],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> List[str]:
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
                    return cast(List[str], [e for _ in inputs])
                else:
                    raise e
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
        inputs: List[LanguageModelInput],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> List[str]:
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
                    return cast(List[str], [e for _ in inputs])
                else:
                    raise e
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
        stop: Optional[List[str]] = None,
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
            callback_manager = CallbackManager.configure(
                config.get("callbacks"),
                self.callbacks,
                self.verbose,
                config.get("tags"),
                self.tags,
                config.get("metadata"),
                self.metadata,
            )
            (run_manager,) = callback_manager.on_llm_start(
                dumpd(self),
                [prompt],
                invocation_params=params,
                options=options,
                name=config.get("run_name"),
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
    ) -> AsyncIterator[str]:
        if type(self)._astream == BaseLLM._astream:
            # model doesn't implement streaming, so use default implementation
            yield await self.ainvoke(input, config=config, stop=stop, **kwargs)
        else:
            prompt = self._convert_input(input).to_string()
            config = ensure_config(config)
            params = self.dict()
            params["stop"] = stop
            params = {**params, **kwargs}
            options = {"stop": stop}
            callback_manager = AsyncCallbackManager.configure(
                config.get("callbacks"),
                self.callbacks,
                self.verbose,
                config.get("tags"),
                self.tags,
                config.get("metadata"),
                self.metadata,
            )
            (run_manager,) = await callback_manager.on_llm_start(
                dumpd(self),
                [prompt],
                invocation_params=params,
                options=options,
                name=config.get("run_name"),
                batch_size=1,
            )
            generation: Optional[GenerationChunk] = None
            try:
                async for chunk in self._astream(
                    prompt, stop=stop, run_manager=run_manager, **kwargs
                ):
                    yield chunk.text
                    if generation is None:
                        generation = chunk
                    else:
                        generation += chunk
                assert generation is not None
            except BaseException as e:
                await run_manager.on_llm_error(
                    e,
                    response=LLMResult(
                        generations=[[generation]] if generation else []
                    ),
                )
                raise e
            else:
                await run_manager.on_llm_end(LLMResult(generations=[[generation]]))

    # --- Custom methods ---

    @abstractmethod
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompts."""

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
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
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        raise NotImplementedError()

    def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        raise NotImplementedError()

    def generate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        callbacks: Optional[Union[Callbacks, List[Callbacks]]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        prompt_strings = [p.to_string() for p in prompts]
        return self.generate(prompt_strings, stop=stop, callbacks=callbacks, **kwargs)

    async def agenerate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        callbacks: Optional[Union[Callbacks, List[Callbacks]]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        prompt_strings = [p.to_string() for p in prompts]
        return await self.agenerate(
            prompt_strings, stop=stop, callbacks=callbacks, **kwargs
        )

    def _generate_helper(
        self,
        prompts: List[str],
        stop: Optional[List[str]],
        run_managers: List[CallbackManagerForLLMRun],
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
            raise e
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
        prompts: List[str],
        stop: Optional[List[str]] = None,
        callbacks: Optional[Union[Callbacks, List[Callbacks]]] = None,
        *,
        tags: Optional[Union[List[str], List[List[str]]]] = None,
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        run_name: Optional[Union[str, List[str]]] = None,
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
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            An LLMResult, which contains a list of candidate Generations for each input
                prompt and additional model provider-specific output.
        """
        if not isinstance(prompts, list):
            raise ValueError(
                "Argument 'prompts' is expected to be of type List[str], received"
                f" argument of type {type(prompts)}."
            )
        # Create callback managers
        if (
            isinstance(callbacks, list)
            and callbacks
            and (
                isinstance(callbacks[0], (list, BaseCallbackManager))
                or callbacks[0] is None
            )
        ):
            # We've received a list of callbacks args to apply to each input
            assert len(callbacks) == len(prompts)
            assert tags is None or (
                isinstance(tags, list) and len(tags) == len(prompts)
            )
            assert metadata is None or (
                isinstance(metadata, list) and len(metadata) == len(prompts)
            )
            assert run_name is None or (
                isinstance(run_name, list) and len(run_name) == len(prompts)
            )
            callbacks = cast(List[Callbacks], callbacks)
            tags_list = cast(List[Optional[List[str]]], tags or ([None] * len(prompts)))
            metadata_list = cast(
                List[Optional[Dict[str, Any]]], metadata or ([{}] * len(prompts))
            )
            run_name_list = run_name or cast(
                List[Optional[str]], ([None] * len(prompts))
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
                    cast(List[str], tags),
                    self.tags,
                    cast(Dict[str, Any], metadata),
                    self.metadata,
                )
            ] * len(prompts)
            run_name_list = [cast(Optional[str], run_name)] * len(prompts)

        params = self.dict()
        params["stop"] = stop
        options = {"stop": stop}
        (
            existing_prompts,
            llm_string,
            missing_prompt_idxs,
            missing_prompts,
        ) = get_prompts(params, prompts)
        disregard_cache = self.cache is not None and not self.cache
        new_arg_supported = inspect.signature(self._generate).parameters.get(
            "run_manager"
        )
        if get_llm_cache() is None or disregard_cache:
            if self.cache is not None and self.cache:
                raise ValueError(
                    "Asked to cache, but no cache found at `langchain.cache`."
                )
            run_managers = [
                callback_manager.on_llm_start(
                    dumpd(self),
                    [prompt],
                    invocation_params=params,
                    options=options,
                    name=run_name,
                    batch_size=len(prompts),
                )[0]
                for callback_manager, prompt, run_name in zip(
                    callback_managers, prompts, run_name_list
                )
            ]
            output = self._generate_helper(
                prompts, stop, run_managers, bool(new_arg_supported), **kwargs
            )
            return output
        if len(missing_prompts) > 0:
            run_managers = [
                callback_managers[idx].on_llm_start(
                    dumpd(self),
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
                existing_prompts, llm_string, missing_prompt_idxs, new_results, prompts
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

    async def _agenerate_helper(
        self,
        prompts: List[str],
        stop: Optional[List[str]],
        run_managers: List[AsyncCallbackManagerForLLMRun],
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
            raise e
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
        prompts: List[str],
        stop: Optional[List[str]] = None,
        callbacks: Optional[Union[Callbacks, List[Callbacks]]] = None,
        *,
        tags: Optional[Union[List[str], List[List[str]]]] = None,
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        run_name: Optional[Union[str, List[str]]] = None,
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
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            An LLMResult, which contains a list of candidate Generations for each input
                prompt and additional model provider-specific output.
        """
        # Create callback managers
        if isinstance(callbacks, list) and (
            isinstance(callbacks[0], (list, BaseCallbackManager))
            or callbacks[0] is None
        ):
            # We've received a list of callbacks args to apply to each input
            assert len(callbacks) == len(prompts)
            assert tags is None or (
                isinstance(tags, list) and len(tags) == len(prompts)
            )
            assert metadata is None or (
                isinstance(metadata, list) and len(metadata) == len(prompts)
            )
            assert run_name is None or (
                isinstance(run_name, list) and len(run_name) == len(prompts)
            )
            callbacks = cast(List[Callbacks], callbacks)
            tags_list = cast(List[Optional[List[str]]], tags or ([None] * len(prompts)))
            metadata_list = cast(
                List[Optional[Dict[str, Any]]], metadata or ([{}] * len(prompts))
            )
            run_name_list = run_name or cast(
                List[Optional[str]], ([None] * len(prompts))
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
                    cast(List[str], tags),
                    self.tags,
                    cast(Dict[str, Any], metadata),
                    self.metadata,
                )
            ] * len(prompts)
            run_name_list = [cast(Optional[str], run_name)] * len(prompts)

        params = self.dict()
        params["stop"] = stop
        options = {"stop": stop}
        (
            existing_prompts,
            llm_string,
            missing_prompt_idxs,
            missing_prompts,
        ) = get_prompts(params, prompts)
        disregard_cache = self.cache is not None and not self.cache
        new_arg_supported = inspect.signature(self._agenerate).parameters.get(
            "run_manager"
        )
        if get_llm_cache() is None or disregard_cache:
            if self.cache is not None and self.cache:
                raise ValueError(
                    "Asked to cache, but no cache found at `langchain.cache`."
                )
            run_managers = await asyncio.gather(
                *[
                    callback_manager.on_llm_start(
                        dumpd(self),
                        [prompt],
                        invocation_params=params,
                        options=options,
                        name=run_name,
                        batch_size=len(prompts),
                    )
                    for callback_manager, prompt, run_name in zip(
                        callback_managers, prompts, run_name_list
                    )
                ]
            )
            run_managers = [r[0] for r in run_managers]
            output = await self._agenerate_helper(
                prompts, stop, run_managers, bool(new_arg_supported), **kwargs
            )
            return output
        if len(missing_prompts) > 0:
            run_managers = await asyncio.gather(
                *[
                    callback_managers[idx].on_llm_start(
                        dumpd(self),
                        [prompts[idx]],
                        invocation_params=params,
                        options=options,
                        name=run_name_list[idx],
                        batch_size=len(missing_prompts),
                    )
                    for idx in missing_prompt_idxs
                ]
            )
            run_managers = [r[0] for r in run_managers]
            new_results = await self._agenerate_helper(
                missing_prompts, stop, run_managers, bool(new_arg_supported), **kwargs
            )
            llm_output = update_cache(
                existing_prompts, llm_string, missing_prompt_idxs, new_results, prompts
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

    @deprecated("0.1.7", alternative="invoke", removal="0.2.0")
    def __call__(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        *,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Check Cache and run the LLM on the given prompt and input."""
        if not isinstance(prompt, str):
            raise ValueError(
                "Argument `prompt` is expected to be a string. Instead found "
                f"{type(prompt)}. If you want to run the LLM on multiple prompts, use "
                "`generate` instead."
            )
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
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        *,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
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

    @deprecated("0.1.7", alternative="invoke", removal="0.2.0")
    def predict(
        self, text: str, *, stop: Optional[Sequence[str]] = None, **kwargs: Any
    ) -> str:
        if stop is None:
            _stop = None
        else:
            _stop = list(stop)
        return self(text, stop=_stop, **kwargs)

    @deprecated("0.1.7", alternative="invoke", removal="0.2.0")
    def predict_messages(
        self,
        messages: List[BaseMessage],
        *,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        text = get_buffer_string(messages)
        if stop is None:
            _stop = None
        else:
            _stop = list(stop)
        content = self(text, stop=_stop, **kwargs)
        return AIMessage(content=content)

    @deprecated("0.1.7", alternative="ainvoke", removal="0.2.0")
    async def apredict(
        self, text: str, *, stop: Optional[Sequence[str]] = None, **kwargs: Any
    ) -> str:
        if stop is None:
            _stop = None
        else:
            _stop = list(stop)
        return await self._call_async(text, stop=_stop, **kwargs)

    @deprecated("0.1.7", alternative="ainvoke", removal="0.2.0")
    async def apredict_messages(
        self,
        messages: List[BaseMessage],
        *,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        text = get_buffer_string(messages)
        if stop is None:
            _stop = None
        else:
            _stop = list(stop)
        content = await self._call_async(text, stop=_stop, **kwargs)
        return AIMessage(content=content)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}

    def __str__(self) -> str:
        """Get a string representation of the object for printing."""
        cls_name = f"\033[1m{self.__class__.__name__}\033[0m"
        return f"{cls_name}\nParams: {self._identifying_params}"

    @property
    @abstractmethod
    def _llm_type(self) -> str:
        """Return type of llm."""

    def dict(self, **kwargs: Any) -> Dict:
        """Return a dictionary of the LLM."""
        starter_dict = dict(self._identifying_params)
        starter_dict["_type"] = self._llm_type
        return starter_dict

    def save(self, file_path: Union[Path, str]) -> None:
        """Save the LLM.

        Args:
            file_path: Path to file to save the LLM to.

        Example:
        .. code-block:: python

            llm.save(file_path="path/llm.yaml")
        """
        # Convert file to Path object.
        if isinstance(file_path, str):
            save_path = Path(file_path)
        else:
            save_path = file_path

        directory_path = save_path.parent
        directory_path.mkdir(parents=True, exist_ok=True)

        # Fetch dictionary to save
        prompt_dict = self.dict()

        if save_path.suffix == ".json":
            with open(file_path, "w") as f:
                json.dump(prompt_dict, f, indent=4)
        elif save_path.suffix == ".yaml":
            with open(file_path, "w") as f:
                yaml.dump(prompt_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"{save_path} must be json or yaml")


class LLM(BaseLLM):
    """Base LLM abstract class.

    The purpose of this class is to expose a simpler interface for working
    with LLMs, rather than expect the user to implement the full _generate method.
    """

    @abstractmethod
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given prompt and input."""

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given prompt and input."""
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
        prompts: List[str],
        stop: Optional[List[str]] = None,
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
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
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
