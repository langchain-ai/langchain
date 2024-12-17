from __future__ import annotations

import asyncio
import functools
import logging
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.utils import get_from_dict_or_env, pre_init
from pydantic import Field
from requests.exceptions import HTTPError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")


def _create_retry_decorator(llm: Tongyi) -> Callable[[Any], Any]:
    min_seconds = 1
    max_seconds = 4
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterward
    return retry(
        reraise=True,
        stop=stop_after_attempt(llm.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(retry_if_exception_type(HTTPError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def check_response(resp: Any) -> Any:
    """Check the response from the completion call."""
    if resp["status_code"] == 200:
        return resp
    elif resp["status_code"] in [400, 401]:
        raise ValueError(
            f"status_code: {resp['status_code']} \n "
            f"code: {resp['code']} \n message: {resp['message']}"
        )
    else:
        raise HTTPError(
            f"HTTP error occurred: status_code: {resp['status_code']} \n "
            f"code: {resp['code']} \n message: {resp['message']}",
            response=resp,
        )


def generate_with_retry(llm: Tongyi, **kwargs: Any) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(llm)

    @retry_decorator
    def _generate_with_retry(**_kwargs: Any) -> Any:
        resp = llm.client.call(**_kwargs)
        return check_response(resp)

    return _generate_with_retry(**kwargs)


def stream_generate_with_retry(llm: Tongyi, **kwargs: Any) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(llm)

    @retry_decorator
    def _stream_generate_with_retry(**_kwargs: Any) -> Any:
        responses = llm.client.call(**_kwargs)
        for resp in responses:
            yield check_response(resp)

    return _stream_generate_with_retry(**kwargs)


async def astream_generate_with_retry(llm: Tongyi, **kwargs: Any) -> Any:
    """Async version of `stream_generate_with_retry`.

    Because the dashscope SDK doesn't provide an async API,
    we wrap `stream_generate_with_retry` with an async generator."""

    class _AioTongyiGenerator:
        def __init__(self, _llm: Tongyi, **_kwargs: Any):
            self.generator = stream_generate_with_retry(_llm, **_kwargs)

        def __aiter__(self) -> AsyncIterator[Any]:
            return self

        async def __anext__(self) -> Any:
            value = await asyncio.get_running_loop().run_in_executor(
                None, self._safe_next
            )
            if value is not None:
                return value
            else:
                raise StopAsyncIteration

        def _safe_next(self) -> Any:
            try:
                return next(self.generator)
            except StopIteration:
                return None

    async for chunk in _AioTongyiGenerator(llm, **kwargs):
        yield chunk


def generate_with_last_element_mark(iterable: Iterable[T]) -> Iterator[Tuple[T, bool]]:
    """Generate elements from an iterable,
    and a boolean indicating if it is the last element."""
    iterator = iter(iterable)
    try:
        item = next(iterator)
    except StopIteration:
        return
    for next_item in iterator:
        yield item, False
        item = next_item
    yield item, True


async def agenerate_with_last_element_mark(
    iterable: AsyncIterable[T],
) -> AsyncIterator[Tuple[T, bool]]:
    """Generate elements from an async iterable,
    and a boolean indicating if it is the last element."""
    iterator = iterable.__aiter__()
    try:
        item = await iterator.__anext__()
    except StopAsyncIteration:
        return
    async for next_item in iterator:
        yield item, False
        item = next_item
    yield item, True


class Tongyi(BaseLLM):
    """Tongyi completion model integration.

    Setup:
        Install ``dashscope`` and set environment variables ``DASHSCOPE_API_KEY``.

        .. code-block:: bash

            pip install dashscope
            export DASHSCOPE_API_KEY="your-api-key"

    Key init args — completion params:
        model: str
            Name of Tongyi model to use.
        top_p: float
            Total probability mass of tokens to consider at each step.
        streaming: bool
            Whether to stream the results or not.

    Key init args — client params:
        api_key: Optional[str]
            Dashscope API KEY. If not passed in will be read from env var DASHSCOPE_API_KEY.
        max_retries: int
            Maximum number of retries to make when generating.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_community.llms import Tongyi

            llm = Tongyi(
                model="qwen-max",
                # top_p="...",
                # api_key="...",
                # other params...
            )

    Invoke:
        .. code-block:: python

            input_text = "用50个字左右阐述，生命的意义在于"
            llm.invoke(input_text)

        .. code-block:: python

            '探索、成长、连接与爱——在有限的时间里，不断学习、体验、贡献并寻找与世界和谐共存之道，让每一刻充满价值与意义。'

    Stream:
        .. code-block:: python

            for chunk in llm.stream(input_text):
                print(chunk)

        .. code-block:: python

            探索 | 、 | 成长 | 、连接与爱。 | 在有限的时间里，寻找个人价值， | 贡献于他人，共同体验世界的美好 | ，让世界因自己的存在而更 | 温暖。

    Async:
        .. code-block:: python

            await llm.ainvoke(input_text)

            # stream:
            # async for chunk in llm.astream(input_text):
            #    print(chunk)

            # batch:
            # await llm.abatch([input_text])

        .. code-block:: python

            '探索、成长、连接与爱。在有限的时间里，寻找个人价值，贡献于他人和社会，体验丰富多彩的情感与经历，不断学习进步，让世界因自己的存在而更美好。'

    """  # noqa: E501

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"dashscope_api_key": "DASHSCOPE_API_KEY"}

    client: Any = None  #: :meta private:
    model_name: str = Field(default="qwen-plus", alias="model")

    """Model name to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    top_p: float = 0.8
    """Total probability mass of tokens to consider at each step."""

    dashscope_api_key: Optional[str] = Field(default=None, alias="api_key")
    """Dashscope api key provide by Alibaba Cloud."""

    streaming: bool = False
    """Whether to stream the results or not."""

    max_retries: int = 10
    """Maximum number of retries to make when generating."""

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "tongyi"

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["dashscope_api_key"] = get_from_dict_or_env(
            values, ["dashscope_api_key", "api_key"], "DASHSCOPE_API_KEY"
        )
        try:
            import dashscope
        except ImportError:
            raise ImportError(
                "Could not import dashscope python package. "
                "Please install it with `pip install dashscope`."
            )
        try:
            values["client"] = dashscope.Generation
        except AttributeError:
            raise ValueError(
                "`dashscope` has no `Generation` attribute, this is likely "
                "due to an old version of the dashscope package. Try upgrading it "
                "with `pip install --upgrade dashscope`."
            )

        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Tongyi Qwen API."""
        normal_params = {
            "model": self.model_name,
            "top_p": self.top_p,
            "api_key": self.dashscope_api_key,
        }

        return {**normal_params, **self.model_kwargs}

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name, **super()._identifying_params}

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        if self.streaming:
            if len(prompts) > 1:
                raise ValueError("Cannot stream results with multiple prompts.")
            generation: Optional[GenerationChunk] = None
            for chunk in self._stream(prompts[0], stop, run_manager, **kwargs):
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            generations.append([self._chunk_to_generation(generation)])
        else:
            params: Dict[str, Any] = self._invocation_params(stop=stop, **kwargs)
            for prompt in prompts:
                completion = generate_with_retry(self, prompt=prompt, **params)
                generations.append(
                    [Generation(**self._generation_from_qwen_resp(completion))]
                )
        return LLMResult(
            generations=generations,
            llm_output={
                "model_name": self.model_name,
            },
        )

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        if self.streaming:
            if len(prompts) > 1:
                raise ValueError("Cannot stream results with multiple prompts.")
            generation: Optional[GenerationChunk] = None
            async for chunk in self._astream(prompts[0], stop, run_manager, **kwargs):
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            generations.append([self._chunk_to_generation(generation)])
        else:
            params: Dict[str, Any] = self._invocation_params(stop=stop, **kwargs)
            for prompt in prompts:
                completion = await asyncio.get_running_loop().run_in_executor(
                    None,
                    functools.partial(
                        generate_with_retry, **{"llm": self, "prompt": prompt, **params}
                    ),
                )
                generations.append(
                    [Generation(**self._generation_from_qwen_resp(completion))]
                )
        return LLMResult(
            generations=generations,
            llm_output={
                "model_name": self.model_name,
            },
        )

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        params: Dict[str, Any] = self._invocation_params(
            stop=stop, stream=True, **kwargs
        )
        for stream_resp, is_last_chunk in generate_with_last_element_mark(
            stream_generate_with_retry(self, prompt=prompt, **params)
        ):
            chunk = GenerationChunk(
                **self._generation_from_qwen_resp(stream_resp, is_last_chunk)
            )
            if run_manager:
                run_manager.on_llm_new_token(
                    chunk.text,
                    chunk=chunk,
                    verbose=self.verbose,
                )
            yield chunk

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        params: Dict[str, Any] = self._invocation_params(
            stop=stop, stream=True, **kwargs
        )
        async for stream_resp, is_last_chunk in agenerate_with_last_element_mark(
            astream_generate_with_retry(self, prompt=prompt, **params)
        ):
            chunk = GenerationChunk(
                **self._generation_from_qwen_resp(stream_resp, is_last_chunk)
            )
            if run_manager:
                await run_manager.on_llm_new_token(
                    chunk.text,
                    chunk=chunk,
                    verbose=self.verbose,
                )
            yield chunk

    def _invocation_params(self, stop: Any, **kwargs: Any) -> Dict[str, Any]:
        params = {
            **self._default_params,
            **kwargs,
        }
        if stop is not None:
            params["stop"] = stop
        if params.get("stream"):
            params["incremental_output"] = True
        return params

    @staticmethod
    def _generation_from_qwen_resp(
        resp: Any, is_last_chunk: bool = True
    ) -> Dict[str, Any]:
        # According to the response from dashscope,
        # each chunk's `generation_info` overwrites the previous one.
        # Besides, The `merge_dicts` method,
        # which is used to concatenate `generation_info` in `GenerationChunk`,
        # does not support merging of int type values.
        # Therefore, we adopt the `generation_info` of the last chunk
        # and discard the `generation_info` of the intermediate chunks.
        if is_last_chunk:
            return dict(
                text=resp["output"]["text"],
                generation_info=dict(
                    finish_reason=resp["output"]["finish_reason"],
                    request_id=resp["request_id"],
                    token_usage=dict(resp["usage"]),
                ),
            )
        else:
            return dict(text=resp["output"]["text"])

    @staticmethod
    def _chunk_to_generation(chunk: GenerationChunk) -> Generation:
        return Generation(
            text=chunk.text,
            generation_info=chunk.generation_info,
        )
