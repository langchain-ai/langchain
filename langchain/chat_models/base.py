import asyncio
import inspect
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pydantic import Extra, Field, root_validator

import langchain
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.manager import (
    AsyncCallbackManager,
    AsyncCallbackManagerForLLMRun,
    CallbackManager,
    CallbackManagerForLLMRun,
    Callbacks,
)
from langchain.load.dump import dumpd
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatResult,
    HumanMessage,
    LLMResult,
    PromptValue,
    RunInfo,
)


def _get_verbosity() -> bool:
    return langchain.verbose


def chat_history_as_string(messages: List[BaseMessage]) -> str:
    return "\n".join([f"{m.type}: {m.content}" for m in messages])


def get_prompts(
    params: Dict[str, Any], prompts: List[List[BaseMessage]]
) -> Tuple[Dict[int, ChatResult], str, List[int], List[List[BaseMessage]]]:
    """Get prompts that are already cached."""
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    missing_prompts = []
    missing_prompt_idxs = []
    existing_prompts = {}
    for i, prompt in enumerate(prompts):
        if langchain.llm_cache is not None:
            cache_val = langchain.llm_cache.lookup(
                chat_history_as_string(prompt), llm_string
            )
            if isinstance(cache_val, list):
                # the cache stores List[Generation]; we parse it to List[ChatGeneration]
                cache_val_for_chat = [
                    ChatGeneration(text=gen.text, message=AIMessage(content=gen.text))
                    for gen in cache_val
                ]

                existing_prompts[i] = ChatResult(generations=cache_val_for_chat)
            else:
                missing_prompts.append(prompt)
                missing_prompt_idxs.append(i)
    return existing_prompts, llm_string, missing_prompt_idxs, missing_prompts


def update_cache(
    existing_prompts: Dict[int, ChatResult],
    llm_string: str,
    missing_prompt_idxs: List[int],
    new_results: List[ChatResult],
    prompts: List[List[BaseMessage]],
) -> List[Optional[dict]]:
    """Update the cache and get the LLM output."""
    for i, result in enumerate(new_results):
        existing_prompts[missing_prompt_idxs[i]] = result
        prompt = prompts[missing_prompt_idxs[i]]
        if langchain.llm_cache is not None:
            langchain.llm_cache.update(
                chat_history_as_string(prompt), llm_string, result.generations
            )
    llm_output = [results.llm_output for results in new_results]
    return llm_output


class BaseChatModel(BaseLanguageModel, ABC):
    cache: Optional[bool] = None
    verbose: bool = Field(default_factory=_get_verbosity)
    """Whether to print out response text."""
    callbacks: Callbacks = Field(default=None, exclude=True)
    callback_manager: Optional[BaseCallbackManager] = Field(default=None, exclude=True)

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

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        """Combine general llm outputs by aggregating them into lists
        e.g. [{"token_usage": 12}, {"token_usage": 18}] -> {"token_usage": [12, 18]}

        Subclasses can override this function if there's a semantically better way to
        combine llm outputs
        e.g. [{"token_usage": 12}, {"token_usage": 18}] -> {"token_usage": 30}
        """
        combined_outputs = {}

        for llm_output in llm_outputs:
            if llm_output is not None:
                for key, value in llm_output.items():
                    if key not in combined_outputs:
                        combined_outputs[key] = [value]
                    else:
                        combined_outputs[key].append(value)

        return combined_outputs

    def generate(
        self,
        messages: List[List[BaseMessage]],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Top Level call"""
        params = self.dict()
        params["stop"] = stop
        (
            existing_prompts,
            llm_string,
            missing_prompt_idxs,
            missing_prompts,
        ) = get_prompts(params, messages)
        disregard_cache = self.cache is not None and not self.cache
        options = {"stop": stop}

        callback_manager = CallbackManager.configure(
            callbacks, self.callbacks, self.verbose
        )
        run_manager = callback_manager.on_chat_model_start(
            dumpd(self), messages, invocation_params=params, options=options
        )
        new_arg_supported = inspect.signature(self._generate).parameters.get(
            "run_manager"
        )
        if langchain.llm_cache is None or disregard_cache:
            # This happens when langchain.cache is None, but self.cache is True
            if self.cache is not None and self.cache:
                raise ValueError(
                    "Asked to cache, but no cache found at `langchain.cache`."
                )
            try:
                results: List[ChatResult] = [
                    self._generate(m, stop=stop, run_manager=run_manager)
                    if new_arg_supported
                    else self._generate(m, stop=stop)
                    for m in messages
                ]
                llm_outputs = [res.llm_output for res in results]
            except (KeyboardInterrupt, Exception) as e:
                run_manager.on_llm_error(e)
                raise e
        else:
            # use cache
            if len(missing_prompts) > 0:
                try:
                    new_results = [
                        self._generate(m, stop=stop, run_manager=run_manager)
                        if new_arg_supported
                        else self._generate(m, stop=stop)
                        for m in missing_prompts
                    ]
                except (KeyboardInterrupt, Exception) as e:
                    run_manager.on_llm_error(e)
                    raise e
                llm_outputs = update_cache(
                    existing_prompts,
                    llm_string,
                    missing_prompt_idxs,
                    new_results,
                    messages,
                )
                # Combine cached results and new results
                results_dict = {
                    **existing_prompts,
                    **dict(zip(missing_prompt_idxs, new_results)),
                }
                results = [result for _, result in sorted(results_dict.items())]
            else:
                llm_outputs = []
                # All prompts were caches, so we construct results solely from cache
                results = [r for _, r in existing_prompts.items()]
        llm_output = self._combine_llm_outputs(llm_outputs)
        generations = [res.generations for res in results]
        # We ignore type as List[List[Generation]] is expected instead of
        # List[List[ChatGeneration]], but ChatGeneration is subclass of Generation
        output = LLMResult(
            generations=generations, llm_output=llm_output  # type: ignore
        )
        run_manager.on_llm_end(output)
        if run_manager:
            output.run = RunInfo(run_id=run_manager.run_id)
        return output

    async def agenerate(
        self,
        messages: List[List[BaseMessage]],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Top Level call"""
        params = self.dict()
        params["stop"] = stop
        (
            existing_prompts,
            llm_string,
            missing_prompt_idxs,
            missing_prompts,
        ) = get_prompts(params, messages)
        disregard_cache = self.cache is not None and not self.cache
        options = {"stop": stop}

        callback_manager = AsyncCallbackManager.configure(
            callbacks, self.callbacks, self.verbose
        )
        run_manager = await callback_manager.on_chat_model_start(
            dumpd(self), messages, invocation_params=params, options=options
        )
        new_arg_supported = inspect.signature(self._agenerate).parameters.get(
            "run_manager"
        )
        if langchain.llm_cache is None or disregard_cache:
            # This happens when langchain.cache is None, but self.cache is True
            if self.cache is not None and self.cache:
                raise ValueError(
                    "Asked to cache, but no cache found at `langchain.cache`."
                )
            try:
                results = await asyncio.gather(
                    *[
                        self._agenerate(m, stop=stop, run_manager=run_manager)
                        if new_arg_supported
                        else self._agenerate(m, stop=stop)
                        for m in messages
                    ]
                )
            except (KeyboardInterrupt, Exception) as e:
                await run_manager.on_llm_error(e)
                raise e
        else:
            # use cache
            if len(missing_prompts) > 0:
                try:
                    new_results = await asyncio.gather(
                        *[
                            self._agenerate(m, stop=stop, run_manager=run_manager)
                            if new_arg_supported
                            else self._agenerate(m, stop=stop)
                            for m in missing_prompts
                        ]
                    )
                except (KeyboardInterrupt, Exception) as e:
                    run_manager.on_llm_error(e)
                    raise e
                llm_outputs = update_cache(
                    existing_prompts,
                    llm_string,
                    missing_prompt_idxs,
                    new_results,
                    messages,
                )
                # Combine cached results and new results
                results_dict = {
                    **existing_prompts,
                    **dict(zip(missing_prompt_idxs, new_results)),
                }
                results = [result for _, result in sorted(results_dict.items())]
            else:
                llm_outputs = []
                # All prompts were caches, so we construct results solely from cache
                results = [r for _, r in existing_prompts.items()]
        llm_output = self._combine_llm_outputs(llm_outputs)
        generations = [res.generations for res in results]
        # We ignore type as List[List[Generation]] is expected instead of
        # List[List[ChatGeneration]], but ChatGeneration is subclass of Generation
        output = LLMResult(
            generations=generations, llm_output=llm_output  # type: ignore
        )
        await run_manager.on_llm_end(output)
        if run_manager:
            output.run = RunInfo(run_id=run_manager.run_id)
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

    @abstractmethod
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call"""

    @abstractmethod
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call"""

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

    def call_as_llm(
        self, message: str, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> str:
        return self.predict(message, stop=stop, **kwargs)

    def predict(
        self, text: str, *, stop: Optional[Sequence[str]] = None, **kwargs: Any
    ) -> str:
        if stop is None:
            _stop = None
        else:
            _stop = list(stop)
        result = self([HumanMessage(content=text)], stop=_stop, **kwargs)
        return result.content

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
        return result.content

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
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}

    @property
    @abstractmethod
    def _llm_type(self) -> str:
        """Return type of chat model."""

    def dict(self, **kwargs: Any) -> Dict:
        """Return a dictionary of the LLM."""
        starter_dict = dict(self._identifying_params)
        starter_dict["_type"] = self._llm_type
        return starter_dict


class SimpleChatModel(BaseChatModel):
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
        func = partial(
            self._generate, messages, stop=stop, run_manager=run_manager, **kwargs
        )
        return await asyncio.get_event_loop().run_in_executor(None, func)
