from __future__ import annotations

import logging
from typing import (
    Any,
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
from langchain_core.pydantic_v1 import Field, root_validator
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


def _create_retry_decorator(llm: LindormAI) -> Callable[[Any], Any]:
    min_seconds = 1
    max_seconds = 4
    return retry(
        reraise=True,
        stop=stop_after_attempt(llm.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(retry_if_exception_type(HTTPError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def generate_with_retry(llm: LindormAI, **kwargs: Any) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(llm)

    @retry_decorator
    def _generate_with_retry(**_kwargs: Any) -> Any:
        resp = llm.client.infer(name=_kwargs["model"], input_data=_kwargs["prompt"], params=_kwargs)
        return resp

    return _generate_with_retry(**kwargs)


def stream_generate_with_retry(llm: LindormAI, **kwargs: Any) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(llm)

    @retry_decorator
    def _stream_generate_with_retry(**_kwargs: Any) -> Any:
        responses = llm.client.stream_infer(name=_kwargs["model"], input_data=_kwargs["prompt"], params=_kwargs)
        for resp in responses:
            yield resp

    return _stream_generate_with_retry(**kwargs)


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


class LindormAILLM(BaseLLM):
    """Lindorm AI large language models.
    To use, you should have the ``lindormai`` python package installed, and the
    necessary credentials for the Lindorm AI endpoint.
    Example:
        .. code-block:: python
            from langchain_community.llms.lindormai import LindormAILLM
            llm = LindormAILLM(
                endpoint='https://ld-xxx-proxy-ml.lindorm.rds.aliyuncs.com:9002',
                username='root',
                password='xxx',
                model_name='qwen-72b'
            )
    """

    client: Any  #: :meta private:
    endpoint: str = Field(...)
    username: str = Field(...)
    password: str = Field(...)
    model_name: str = Field(...)
    """Model name to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    temperature: float = 0.8
    """Temperature to use during generation."""
    streaming: bool = False
    """Whether to stream the results or not."""
    max_retries: int = 10
    """Maximum number of retries to make when generating."""

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "lindormai"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate the environment settings."""
        try:
            import lindormai
        except ImportError:
            raise ImportError(
                "Could not import lindormai python package. "
                "Please install it with `pip install lindormai`."
            )
        try:
            from lindormai.model_manager import ModelManager
            values["client"] = ModelManager(values['endpoint'], values['username'], values['password'])
        except AttributeError:
            raise ValueError(
                "`lindormai` does not have `ModelManager` attribute, please check version."
            )
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Lindorm AI API."""
        normal_params = {
            "model": self.model_name,
            "temperature": self.temperature,
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
                    [Generation(**self._generation_from_lindormai_resp(completion))]
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
        raise NotImplementedError("Please use `_generate`. Official does not support asynchronous requests")

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
                **self._generation_from_lindormai_resp(stream_resp, is_last_chunk)
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
        raise NotImplementedError("Please use `_stream`. Official does not support asynchronous requests")

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
    def _generation_from_lindormai_resp(
            resp: Any, is_last_chunk: bool = True
    ) -> Dict[str, Any]:
        if resp is None:
            raise ValueError("The response is None. Expected a non-None value.")
        elif 'output' in resp:
            content = resp['output']
        elif 'outputs' in resp:
            content = ''
            for output in resp['outputs']:
                content += output
        else:
            content = resp
        return dict(text=content)

    @staticmethod
    def _chunk_to_generation(chunk: GenerationChunk) -> Generation:
        return Generation(
            text=chunk.text,
            generation_info=chunk.generation_info,
        )
