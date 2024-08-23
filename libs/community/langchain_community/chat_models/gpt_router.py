from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.messages import AIMessageChunk, BaseMessage, BaseMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env

from langchain_community.adapters.openai import (
    convert_dict_to_message,
    convert_message_to_dict,
)
from langchain_community.chat_models.openai import _convert_delta_to_message_chunk

if TYPE_CHECKING:
    from gpt_router.models import ChunkedGenerationResponse, GenerationResponse


logger = logging.getLogger(__name__)

DEFAULT_API_BASE_URL = "https://gpt-router-preview.writesonic.com"


class GPTRouterException(Exception):
    """Error with the `GPTRouter APIs`"""


class GPTRouterModel(BaseModel):
    """GPTRouter model."""

    name: str
    provider_name: str


def get_ordered_generation_requests(
    models_priority_list: List[GPTRouterModel], **kwargs: Any
) -> List:
    """
    Return the body for the model router input.
    """

    from gpt_router.models import GenerationParams, ModelGenerationRequest

    return [
        ModelGenerationRequest(
            model_name=model.name,
            provider_name=model.provider_name,
            order=index + 1,
            prompt_params=GenerationParams(**kwargs),
        )
        for index, model in enumerate(models_priority_list)
    ]


def _create_retry_decorator(
    llm: GPTRouter,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:
    from gpt_router import exceptions

    errors = [
        exceptions.GPTRouterApiTimeoutError,
        exceptions.GPTRouterInternalServerError,
        exceptions.GPTRouterNotAvailableError,
        exceptions.GPTRouterTooManyRequestsError,
    ]
    return create_base_retry_decorator(
        error_types=errors, max_retries=llm.max_retries, run_manager=run_manager
    )


def completion_with_retry(
    llm: GPTRouter,
    models_priority_list: List[GPTRouterModel],
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Union[GenerationResponse, Generator[ChunkedGenerationResponse, None, None]]:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    def _completion_with_retry(**kwargs: Any) -> Any:
        ordered_generation_requests = get_ordered_generation_requests(
            models_priority_list, **kwargs
        )
        return llm.client.generate(
            ordered_generation_requests=ordered_generation_requests,
            is_stream=kwargs.get("stream", False),
        )

    return _completion_with_retry(**kwargs)


async def acompletion_with_retry(
    llm: GPTRouter,
    models_priority_list: List[GPTRouterModel],
    run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Union[GenerationResponse, AsyncGenerator[ChunkedGenerationResponse, None]]:
    """Use tenacity to retry the async completion call."""

    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        ordered_generation_requests = get_ordered_generation_requests(
            models_priority_list, **kwargs
        )
        return await llm.client.agenerate(
            ordered_generation_requests=ordered_generation_requests,
            is_stream=kwargs.get("stream", False),
        )

    return await _completion_with_retry(**kwargs)


class GPTRouter(BaseChatModel):
    """GPTRouter by Writesonic Inc.

    For more information, see https://gpt-router.writesonic.com/docs
    """

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    models_priority_list: List[GPTRouterModel] = Field(min_items=1)
    gpt_router_api_base: str = Field(default=None)
    """WriteSonic GPTRouter custom endpoint"""
    gpt_router_api_key: Optional[SecretStr] = None
    """WriteSonic GPTRouter API Key"""
    temperature: float = 0.7
    """What sampling temperature to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    max_retries: int = 4
    """Maximum number of retries to make when generating."""
    streaming: bool = False
    """Whether to stream the results or not."""
    n: int = 1
    """Number of chat completions to generate for each prompt."""
    max_tokens: int = 256

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        values["gpt_router_api_base"] = get_from_dict_or_env(
            values,
            "gpt_router_api_base",
            "GPT_ROUTER_API_BASE",
            DEFAULT_API_BASE_URL,
        )

        values["gpt_router_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(
                values,
                "gpt_router_api_key",
                "GPT_ROUTER_API_KEY",
            )
        )
        return values

    @root_validator(pre=True, skip_on_failure=True)
    def post_init(cls, values: Dict) -> Dict:
        try:
            from gpt_router.client import GPTRouterClient

        except ImportError:
            raise GPTRouterException(
                "Could not import GPTRouter python package. "
                "Please install it with `pip install GPTRouter`."
            )

        gpt_router_client = GPTRouterClient(
            values["gpt_router_api_base"],
            values["gpt_router_api_key"].get_secret_value(),
        )
        values["client"] = gpt_router_client

        return values

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"gpt_router_api_key": "GPT_ROUTER_API_KEY"}

    @property
    def lc_serializable(self) -> bool:
        return True

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "gpt-router-chat"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"models_priority_list": self.models_priority_list},
            **self._default_params,
        }

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling GPTRouter API."""
        return {
            "max_tokens": self.max_tokens,
            "stream": self.streaming,
            "n": self.n,
            "temperature": self.temperature,
            **self.model_kwargs,
        }

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": False}
        response = completion_with_retry(
            self,
            messages=message_dicts,
            models_priority_list=self.models_priority_list,
            run_manager=run_manager,
            **params,
        )
        return self._create_chat_result(response)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": False}
        response = await acompletion_with_retry(
            self,
            messages=message_dicts,
            models_priority_list=self.models_priority_list,
            run_manager=run_manager,
            **params,
        )
        return self._create_chat_result(response)

    def _create_chat_generation_chunk(
        self, data: Mapping[str, Any], default_chunk_class: Type[BaseMessageChunk]
    ) -> Tuple[ChatGenerationChunk, Type[BaseMessageChunk]]:
        chunk = _convert_delta_to_message_chunk(
            {"content": data.get("text", "")}, default_chunk_class
        )
        finish_reason = data.get("finish_reason")
        generation_info = (
            dict(finish_reason=finish_reason) if finish_reason is not None else None
        )
        default_chunk_class = chunk.__class__
        gen_chunk = ChatGenerationChunk(message=chunk, generation_info=generation_info)
        return gen_chunk, default_chunk_class

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class: Type[BaseMessageChunk] = AIMessageChunk
        generator_response = completion_with_retry(
            self,
            messages=message_dicts,
            models_priority_list=self.models_priority_list,
            run_manager=run_manager,
            **params,
        )
        for chunk in generator_response:
            if chunk.event != "update":
                continue

            chunk, default_chunk_class = self._create_chat_generation_chunk(
                chunk.data, default_chunk_class
            )

            if run_manager:
                run_manager.on_llm_new_token(
                    token=chunk.message.content, chunk=chunk.message
                )

            yield chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class: Type[BaseMessageChunk] = AIMessageChunk
        generator_response = acompletion_with_retry(
            self,
            messages=message_dicts,
            models_priority_list=self.models_priority_list,
            run_manager=run_manager,
            **params,
        )
        async for chunk in await generator_response:
            if chunk.event != "update":
                continue

            chunk, default_chunk_class = self._create_chat_generation_chunk(
                chunk.data, default_chunk_class
            )

            if run_manager:
                await run_manager.on_llm_new_token(
                    token=chunk.message.content, chunk=chunk.message
                )

            yield chunk

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = self._default_params
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _create_chat_result(self, response: GenerationResponse) -> ChatResult:
        generations = []
        for res in response.choices:
            message = convert_dict_to_message(
                {
                    "role": "assistant",
                    "content": res.text,
                }
            )
            gen = ChatGeneration(
                message=message,
                generation_info=dict(finish_reason=res.finish_reason),
            )
            generations.append(gen)
        llm_output = {"token_usage": response.meta, "model": response.model}
        return ChatResult(generations=generations, llm_output=llm_output)
