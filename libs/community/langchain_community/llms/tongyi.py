from __future__ import annotations

import asyncio
import functools
import logging
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_from_dict_or_env
from requests.exceptions import HTTPError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


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
    if resp.status_code == 200:
        return resp
    elif resp.status_code in [400, 401]:
        raise ValueError(
            f"status_code: {resp.status_code} \n "
            f"code: {resp.code} \n message: {resp.message}"
        )
    else:
        raise HTTPError(
            f"HTTP error occurred: status_code: {resp.status_code} \n "
            f"code: {resp.code} \n message: {resp.message}",
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


class Tongyi(BaseLLM):
    """Tongyi Qwen large language models.

    To use, you should have the ``dashscope`` python package installed, and the
    environment variable ``DASHSCOPE_API_KEY`` set with your API key, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_community.llms import Tongyi
            tongyi = tongyi()
    """

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"dashscope_api_key": "DASHSCOPE_API_KEY"}

    client: Any  #: :meta private:
    model_name: str = "qwen-plus"

    """Model name to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    top_p: float = 0.8
    """Total probability mass of tokens to consider at each step."""

    dashscope_api_key: Optional[str] = None
    """Dashscope api key provide by Alibaba Cloud."""

    streaming: bool = False
    """Whether to stream the results or not."""

    max_retries: int = 10
    """Maximum number of retries to make when generating."""

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "tongyi"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["dashscope_api_key"] = get_from_dict_or_env(
            values, "dashscope_api_key", "DASHSCOPE_API_KEY"
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
        for stream_resp in stream_generate_with_retry(self, prompt=prompt, **params):
            chunk = GenerationChunk(**self._generation_from_qwen_resp(stream_resp))
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
        async for stream_resp in astream_generate_with_retry(
            self, prompt=prompt, **params
        ):
            chunk = GenerationChunk(**self._generation_from_qwen_resp(stream_resp))
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
    def _generation_from_qwen_resp(resp: Any) -> Dict[str, Any]:
        return dict(
            text=resp["output"]["text"],
            generation_info=dict(
                finish_reason=resp["output"]["finish_reason"],
                request_id=resp["request_id"],
                token_usage=dict(resp["usage"]),
            ),
        )

    @staticmethod
    def _chunk_to_generation(chunk: GenerationChunk) -> Generation:
        return Generation(
            text=chunk.text,
            generation_info=chunk.generation_info,
        )
