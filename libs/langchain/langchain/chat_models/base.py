import asyncio
import inspect
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Union,
    cast,
)

from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.manager import (
    AsyncCallbackManager,
    AsyncCallbackManagerForLLMRun,
    CallbackManager,
    CallbackManagerForLLMRun,
    Callbacks,
)
from langchain.globals import get_llm_cache
from langchain.load.dump import dumpd, dumps
from langchain.prompts.base import StringPromptValue
from langchain.prompts.chat import ChatPromptValue
from langchain.pydantic_v1 import Field, root_validator
from langchain.schema import (
    ChatGeneration,
    ChatResult,
    LLMResult,
    PromptValue,
    RunInfo,
)
from langchain.schema.language_model import BaseLanguageModel, LanguageModelInput
from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.schema.output import ChatGenerationChunk
from langchain.schema.runnable import RunnableConfig


def _get_verbosity() -> bool:
    from langchain.globals import get_verbose

    return get_verbose()


def _generate_from_stream(stream: Iterator[ChatGenerationChunk]) -> ChatResult:
    generation: Optional[ChatGenerationChunk] = None
    for chunk in stream:
        if generation is None:
            generation = chunk
        else:
            generation += chunk
    assert generation is not None
    return ChatResult(generations=[generation])


async def _agenerate_from_stream(
    stream: AsyncIterator[ChatGenerationChunk],
) -> ChatResult:
    generation: Optional[ChatGenerationChunk] = None
    async for chunk in stream:
        if generation is None:
            generation = chunk
        else:
            generation += chunk
    assert generation is not None
    return ChatResult(generations=[generation])


class BaseChatModel(BaseLanguageModel[BaseMessageChunk], ABC):
    """Base class for Chat models."""

    cache: Optional[bool] = None
    """Whether to cache the response."""
    verbose: bool = Field(default_factory=_get_verbosity)
    """Whether to print out response text."""
    callbacks: Callbacks = Field(default=None, exclude=True)
    """Callbacks to add to the run trace."""
    callback_manager: Optional[BaseCallbackManager] = Field(default=None, exclude=True)
    """Callback manager to add to the run trace."""
    tags: Optional[List[str]] = Field(default=None, exclude=True)
    """Tags to add to the run trace."""
    metadata: Optional[Dict[str, Any]] = Field(default=None, exclude=True)
    """Metadata to add to the run trace."""

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

        arbitrary_types_allowed = True

    # --- Runnable methods ---

    @property
    def OutputType(self) -> Any:
        """Get the output type for this runnable."""
        return Union[
            HumanMessage, AIMessage, ChatMessage, FunctionMessage, SystemMessage
        ]

    def _convert_input(self, input: LanguageModelInput) -> PromptValue:
        if isinstance(input, PromptValue):
            return input
        elif isinstance(input, str):
            return StringPromptValue(text=input)
        elif isinstance(input, list):
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
    ) -> BaseMessageChunk:
        config = config or {}
        return cast(
            BaseMessageChunk,
            cast(
                ChatGeneration,
                self.generate_prompt(
                    [self._convert_input(input)],
                    stop=stop,
                    callbacks=config.get("callbacks"),
                    tags=config.get("tags"),
                    metadata=config.get("metadata"),
                    run_name=config.get("run_name"),
                    **kwargs,
                ).generations[0][0],
            ).message,
        )

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> BaseMessageChunk:
        config = config or {}
        llm_result = await self.agenerate_prompt(
            [self._convert_input(input)],
            stop=stop,
            callbacks=config.get("callbacks"),
            tags=config.get("tags"),
            metadata=config.get("metadata"),
            run_name=config.get("run_name"),
            **kwargs,
        )
        return cast(
            BaseMessageChunk, cast(ChatGeneration, llm_result.generations[0][0]).message
        )

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
            yield self.invoke(input, config=config, stop=stop, **kwargs)
        else:
            config = config or {}
            messages = self._convert_input(input).to_messages()
            params = self._get_invocation_params(stop=stop, **kwargs)
            options = {"stop": stop, **kwargs}
            callback_manager = CallbackManager.configure(
                config.get("callbacks"),
                self.callbacks,
                self.verbose,
                config.get("tags"),
                self.tags,
                config.get("metadata"),
                self.metadata,
            )
            (run_manager,) = callback_manager.on_chat_model_start(
                dumpd(self),
                [messages],
                invocation_params=params,
                options=options,
                name=config.get("run_name"),
            )
            try:
                generation: Optional[ChatGenerationChunk] = None
                for chunk in self._stream(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                ):
                    yield chunk.message
                    if generation is None:
                        generation = chunk
                    else:
                        generation += chunk
                assert generation is not None
            except BaseException as e:
                run_manager.on_llm_error(e)
                raise e
            else:
                run_manager.on_llm_end(
                    LLMResult(generations=[[generation]]),
                )

    async def astream(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[BaseMessageChunk]:
        if type(self)._astream == BaseChatModel._astream:
            # model doesn't implement streaming, so use default implementation
            yield self.invoke(input, config=config, stop=stop, **kwargs)
        else:
            config = config or {}
            messages = self._convert_input(input).to_messages()
            params = self._get_invocation_params(stop=stop, **kwargs)
            options = {"stop": stop, **kwargs}
            callback_manager = AsyncCallbackManager.configure(
                config.get("callbacks"),
                self.callbacks,
                self.verbose,
                config.get("tags"),
                self.tags,
                config.get("metadata"),
                self.metadata,
            )
            (run_manager,) = await callback_manager.on_chat_model_start(
                dumpd(self),
                [messages],
                invocation_params=params,
                options=options,
                name=config.get("run_name"),
            )
            try:
                generation: Optional[ChatGenerationChunk] = None
                async for chunk in self._astream(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                ):
                    yield chunk.message
                    if generation is None:
                        generation = chunk
                    else:
                        generation += chunk
                assert generation is not None
            except BaseException as e:
                await run_manager.on_llm_error(e)
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

    def _get_llm_string(self, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        if self.is_lc_serializable():
            params = {**kwargs, **{"stop": stop}}
            param_string = str(sorted([(k, v) for k, v in params.items()]))
            llm_string = dumps(self)
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
        **kwargs: Any,
    ) -> LLMResult:
        """Top Level call"""
        params = self._get_invocation_params(stop=stop, **kwargs)
        options = {"stop": stop}

        callback_manager = CallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose,
            tags,
            self.tags,
            metadata,
            self.metadata,
        )
        run_managers = callback_manager.on_chat_model_start(
            dumpd(self),
            messages,
            invocation_params=params,
            options=options,
            name=run_name,
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
                    run_managers[i].on_llm_error(e)
                raise e
        flattened_outputs = [
            LLMResult(generations=[res.generations], llm_output=res.llm_output)
            for res in results
        ]
        llm_output = self._combine_llm_outputs([res.llm_output for res in results])
        generations = [res.generations for res in results]
        output = LLMResult(generations=generations, llm_output=llm_output)
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
        **kwargs: Any,
    ) -> LLMResult:
        """Top Level call"""
        params = self._get_invocation_params(stop=stop, **kwargs)
        options = {"stop": stop}

        callback_manager = AsyncCallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose,
            tags,
            self.tags,
            metadata,
            self.metadata,
        )

        run_managers = await callback_manager.on_chat_model_start(
            dumpd(self),
            messages,
            invocation_params=params,
            options=options,
            name=run_name,
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
                    await run_managers[i].on_llm_error(res)
                exceptions.append(res)
        if exceptions:
            if run_managers:
                await asyncio.gather(
                    *[
                        run_manager.on_llm_end(
                            LLMResult(
                                generations=[res.generations], llm_output=res.llm_output
                            )
                        )
                        for run_manager, res in zip(run_managers, results)
                        if not isinstance(res, Exception)
                    ]
                )
            raise exceptions[0]
        flattened_outputs = [
            LLMResult(generations=[res.generations], llm_output=res.llm_output)
            for res in results
        ]
        llm_output = self._combine_llm_outputs([res.llm_output for res in results])
        generations = [res.generations for res in results]
        output = LLMResult(generations=generations, llm_output=llm_output)
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
        new_arg_supported = inspect.signature(self._generate).parameters.get(
            "run_manager"
        )
        disregard_cache = self.cache is not None and not self.cache
        llm_cache = get_llm_cache()
        if llm_cache is None or disregard_cache:
            # This happens when langchain.cache is None, but self.cache is True
            if self.cache is not None and self.cache:
                raise ValueError(
                    "Asked to cache, but no cache found at `langchain.cache`."
                )
            if new_arg_supported:
                return self._generate(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
            else:
                return self._generate(messages, stop=stop, **kwargs)
        else:
            llm_string = self._get_llm_string(stop=stop, **kwargs)
            prompt = dumps(messages)
            cache_val = llm_cache.lookup(prompt, llm_string)
            if isinstance(cache_val, list):
                return ChatResult(generations=cache_val)
            else:
                if new_arg_supported:
                    result = self._generate(
                        messages, stop=stop, run_manager=run_manager, **kwargs
                    )
                else:
                    result = self._generate(messages, stop=stop, **kwargs)
                llm_cache.update(prompt, llm_string, result.generations)
                return result

    async def _agenerate_with_cache(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        new_arg_supported = inspect.signature(self._agenerate).parameters.get(
            "run_manager"
        )
        disregard_cache = self.cache is not None and not self.cache
        llm_cache = get_llm_cache()
        if llm_cache is None or disregard_cache:
            # This happens when langchain.cache is None, but self.cache is True
            if self.cache is not None and self.cache:
                raise ValueError(
                    "Asked to cache, but no cache found at `langchain.cache`."
                )
            if new_arg_supported:
                return await self._agenerate(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
            else:
                return await self._agenerate(messages, stop=stop, **kwargs)
        else:
            llm_string = self._get_llm_string(stop=stop, **kwargs)
            prompt = dumps(messages)
            cache_val = llm_cache.lookup(prompt, llm_string)
            if isinstance(cache_val, list):
                return ChatResult(generations=cache_val)
            else:
                if new_arg_supported:
                    result = await self._agenerate(
                        messages, stop=stop, run_manager=run_manager, **kwargs
                    )
                else:
                    result = await self._agenerate(messages, stop=stop, **kwargs)
                llm_cache.update(prompt, llm_string, result.generations)
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
        return await asyncio.get_running_loop().run_in_executor(
            None, partial(self._generate, **kwargs), messages, stop, run_manager
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        raise NotImplementedError()

    def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        raise NotImplementedError()

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
    def _identifying_params(self) -> Dict[str, Any]:
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
    """Simple Chat Model."""

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
