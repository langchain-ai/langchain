import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Type,
    Union,
)

from langchain.adapters.openai import convert_message_to_dict
from langchain.callbacks.base import Callbacks
from langchain.callbacks.manager import (
    AsyncCallbackManager,
    AsyncCallbackManagerForLLMRun,
    CallbackManager,
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import create_base_retry_decorator
from langchain.load.dump import dumpd
from langchain.pydantic_v1 import Field, root_validator
from langchain.schema.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
)
from langchain.schema.output import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    LLMResult,
    RunInfo,
)
from langchain.utils.env import get_from_dict_or_env


def _convert_delta_to_message_chunk(
    _dict: Any, default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    """Convert a delta response to a message chunk."""
    role = _dict.role
    content = _dict.content or ""
    additional_kwargs: Dict = {}

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    elif role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict.name)
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)
    else:
        return default_class(type="", content=content)


def convert_dict_to_message(_dict: Any) -> BaseMessage:
    """Convert a dict response to a message."""
    role = _dict.role
    content = _dict.content or ""
    if role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        content = _dict.content
        additional_kwargs: Dict = {}
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    elif role == "system":
        return SystemMessage(content=content)
    elif role == "function":
        return FunctionMessage(content=content, name=_dict.name)
    else:
        return ChatMessage(content=content, role=role)


class ChatFireworks(BaseChatModel):
    """Fireworks Chat models."""

    model: str = "accounts/fireworks/models/llama-v2-7b-chat"
    model_kwargs: dict = Field(
        default_factory=lambda: {
            "temperature": 0.7,
            "max_tokens": 512,
            "top_p": 1,
        }.copy()
    )
    fireworks_api_key: Optional[str] = None
    max_retries: int = 20
    batch_size: int = 20

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"fireworks_api_key": "FIREWORKS_API_KEY"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key in environment."""
        try:
            import fireworks.client
        except ImportError as e:
            raise ImportError(
                "Could not import fireworks-ai python package. "
                "Please install it with `pip install fireworks-ai`."
            ) from e
        fireworks_api_key = get_from_dict_or_env(
            values, "fireworks_api_key", "FIREWORKS_API_KEY"
        )
        fireworks.client.api_key = fireworks_api_key
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fireworks-chat"

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

        def _completion_with_retry_batching(
            message: List[List[BaseMessage]],
        ) -> List[Any]:
            args_list = [
                (m, stop, run_managers[i] if run_managers else None, kwargs)
                for i, m in enumerate(message)
            ]
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(self._process_message, args_list))

            return results

        sub_messages = self.get_batch_messages(params, messages, stop)

        results = []
        for message in sub_messages:
            results.extend(_completion_with_retry_batching(message))

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

        async def _acompletion_with_retry_batching(
            message: List[List[BaseMessage]],
        ) -> List[Any]:
            args_list = [
                (m, stop, run_managers[i] if run_managers else None, kwargs)
                for i, m in enumerate(message)
            ]
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                results = await asyncio.gather(
                    *[
                        loop.run_in_executor(executor, self._process_message, args)
                        for args in args_list
                    ],
                    return_exceptions=True,
                )

            return results

        sub_messages = self.get_batch_messages(params, messages, stop)

        results = []
        for message in sub_messages:
            results.extend(await _acompletion_with_retry_batching(message))

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

    def _process_message(self, args: List[Any]) -> ChatResult:
        m, stop, run_manager, kwargs = args
        try:
            return self._generate_with_cache(
                messages=m, stop=stop, run_manager=run_manager, **kwargs
            )
        except BaseException as e:
            if run_manager:
                run_manager.on_llm_error(e)
            raise e

    def get_batch_messages(
        self,
        params: Dict[str, Any],
        messages: List[List[BaseMessage]],
        stop: Optional[List[str]] = None,
    ) -> List[List[List[BaseMessage]]]:
        """Get the sub messages for llm call."""
        if stop is not None:
            params["stop"] = stop

        sub_messages = [
            messages[i : i + self.batch_size]
            for i in range(0, len(messages), self.batch_size)
        ]
        return sub_messages

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message_dicts = self._create_message_dicts(messages)

        params = {
            "model": self.model,
            "messages": message_dicts,
            **self.model_kwargs,
        }
        response = completion_with_retry(
            self, run_manager=run_manager, stop=stop, **params
        )
        return self._create_chat_result(response)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message_dicts = self._create_message_dicts(messages)
        params = {
            "model": self.model,
            "messages": message_dicts,
            **self.model_kwargs,
        }
        response = await acompletion_with_retry(
            self, run_manager=run_manager, stop=stop, **params
        )
        return self._create_chat_result(response)

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        if llm_outputs[0] is None:
            return {}
        return llm_outputs[0]

    def _create_chat_result(self, response: Any) -> ChatResult:
        generations = []
        for res in response.choices:
            message = convert_dict_to_message(res.message)
            gen = ChatGeneration(
                message=message,
                generation_info=dict(finish_reason=res.finish_reason),
            )
            generations.append(gen)
        llm_output = {"model": self.model}
        return ChatResult(generations=generations, llm_output=llm_output)

    def _create_message_dicts(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, Any]]:
        message_dicts = [convert_message_to_dict(m) for m in messages]
        return message_dicts

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts = self._create_message_dicts(messages)
        default_chunk_class = AIMessageChunk
        params = {
            "model": self.model,
            "messages": message_dicts,
            "stream": True,
            **self.model_kwargs,
        }
        for chunk in completion_with_retry(
            self, run_manager=run_manager, stop=stop, **params
        ):
            choice = chunk.choices[0]
            chunk = _convert_delta_to_message_chunk(choice.delta, default_chunk_class)
            finish_reason = choice.finish_reason
            generation_info = (
                dict(finish_reason=finish_reason) if finish_reason is not None else None
            )
            default_chunk_class = chunk.__class__
            yield ChatGenerationChunk(message=chunk, generation_info=generation_info)
            if run_manager:
                run_manager.on_llm_new_token(chunk.content, chunk=chunk)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts = self._create_message_dicts(messages)
        default_chunk_class = AIMessageChunk
        params = {
            "model": self.model,
            "messages": message_dicts,
            "stream": True,
            **self.model_kwargs,
        }
        async for chunk in await acompletion_with_retry_streaming(
            self, run_manager=run_manager, stop=stop, **params
        ):
            choice = chunk.choices[0]
            chunk = _convert_delta_to_message_chunk(choice.delta, default_chunk_class)
            finish_reason = choice.finish_reason
            generation_info = (
                dict(finish_reason=finish_reason) if finish_reason is not None else None
            )
            default_chunk_class = chunk.__class__
            yield ChatGenerationChunk(message=chunk, generation_info=generation_info)
            if run_manager:
                await run_manager.on_llm_new_token(token=chunk.content, chunk=chunk)


def completion_with_retry(
    llm: ChatFireworks,
    *,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    import fireworks.client

    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    def _completion_with_retry(**kwargs: Any) -> Any:
        return fireworks.client.ChatCompletion.create(
            **kwargs,
        )

    return _completion_with_retry(**kwargs)


async def acompletion_with_retry(
    llm: ChatFireworks,
    *,
    run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the async completion call."""
    import fireworks.client

    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        return await fireworks.client.ChatCompletion.acreate(
            **kwargs,
        )

    return await _completion_with_retry(**kwargs)


async def acompletion_with_retry_streaming(
    llm: ChatFireworks,
    *,
    run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call for streaming."""
    import fireworks.client

    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        return fireworks.client.ChatCompletion.acreate(
            **kwargs,
        )

    return await _completion_with_retry(**kwargs)


def _create_retry_decorator(
    llm: ChatFireworks,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:
    """Define retry mechanism."""
    import fireworks.client

    errors = [
        fireworks.client.error.RateLimitError,
        fireworks.client.error.ServiceUnavailableError,
    ]
    return create_base_retry_decorator(
        error_types=errors, max_retries=llm.max_retries, run_manager=run_manager
    )
