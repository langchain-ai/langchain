import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Union

from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import BaseLLM, create_base_retry_decorator
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.utils import convert_to_secret_str, pre_init
from langchain_core.utils.env import get_from_dict_or_env
from pydantic import Field, SecretStr


def _stream_response_to_generation_chunk(
    stream_response: Any,
) -> GenerationChunk:
    """Convert a stream response to a generation chunk."""
    return GenerationChunk(
        text=stream_response.choices[0].text,
        generation_info=dict(
            finish_reason=stream_response.choices[0].finish_reason,
            logprobs=stream_response.choices[0].logprobs,
        ),
    )


@deprecated(
    since="0.0.26",
    removal="1.0",
    alternative_import="langchain_fireworks.Fireworks",
)
class Fireworks(BaseLLM):
    """Fireworks models."""

    model: str = "accounts/fireworks/models/llama-v2-7b-chat"
    model_kwargs: dict = Field(
        default_factory=lambda: {
            "temperature": 0.7,
            "max_tokens": 512,
            "top_p": 1,
        }.copy()
    )
    fireworks_api_key: Optional[SecretStr] = None
    max_retries: int = 20
    batch_size: int = 20
    use_retry: bool = True

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"fireworks_api_key": "FIREWORKS_API_KEY"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "llms", "fireworks"]

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key in environment."""
        try:
            import fireworks.client
        except ImportError as e:
            raise ImportError(
                "Could not import fireworks-ai python package. "
                "Please install it with `pip install fireworks-ai`."
            ) from e
        fireworks_api_key = convert_to_secret_str(
            get_from_dict_or_env(values, "fireworks_api_key", "FIREWORKS_API_KEY")
        )
        fireworks.client.api_key = fireworks_api_key.get_secret_value()
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fireworks"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Call out to Fireworks endpoint with k unique prompts.
        Args:
            prompts: The prompts to pass into the model.
            stop: Optional list of stop words to use when generating.
        Returns:
            The full LLM output.
        """
        params = {
            "model": self.model,
            **self.model_kwargs,
        }
        sub_prompts = self.get_batch_prompts(prompts)
        choices = []
        for _prompts in sub_prompts:
            response = completion_with_retry_batching(
                self,
                self.use_retry,
                prompt=_prompts,
                run_manager=run_manager,
                stop=stop,
                **params,
            )
            choices.extend(response)

        return self.create_llm_result(choices, prompts)

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Call out to Fireworks endpoint async with k unique prompts."""
        params = {
            "model": self.model,
            **self.model_kwargs,
        }
        sub_prompts = self.get_batch_prompts(prompts)
        choices = []
        for _prompts in sub_prompts:
            response = await acompletion_with_retry_batching(
                self,
                self.use_retry,
                prompt=_prompts,
                run_manager=run_manager,
                stop=stop,
                **params,
            )
            choices.extend(response)

        return self.create_llm_result(choices, prompts)

    def get_batch_prompts(
        self,
        prompts: List[str],
    ) -> List[List[str]]:
        """Get the sub prompts for llm call."""
        sub_prompts = [
            prompts[i : i + self.batch_size]
            for i in range(0, len(prompts), self.batch_size)
        ]
        return sub_prompts

    def create_llm_result(self, choices: Any, prompts: List[str]) -> LLMResult:
        """Create the LLMResult from the choices and prompts."""
        generations = []
        for i, _ in enumerate(prompts):
            sub_choices = choices[i : (i + 1)]
            generations.append(
                [
                    Generation(
                        text=choice.__dict__["choices"][0].text,
                    )
                    for choice in sub_choices
                ]
            )
        llm_output = {"model": self.model}
        return LLMResult(generations=generations, llm_output=llm_output)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        params = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            **self.model_kwargs,
        }
        for stream_resp in completion_with_retry(
            self, self.use_retry, run_manager=run_manager, stop=stop, **params
        ):
            chunk = _stream_response_to_generation_chunk(stream_resp)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        params = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            **self.model_kwargs,
        }
        async for stream_resp in await acompletion_with_retry_streaming(
            self, self.use_retry, run_manager=run_manager, stop=stop, **params
        ):
            chunk = _stream_response_to_generation_chunk(stream_resp)
            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk


def conditional_decorator(
    condition: bool, decorator: Callable[[Any], Any]
) -> Callable[[Any], Any]:
    """Conditionally apply a decorator.

    Args:
        condition: A boolean indicating whether to apply the decorator.
        decorator: A decorator function.

    Returns:
        A decorator function.
    """

    def actual_decorator(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
        if condition:
            return decorator(func)
        return func

    return actual_decorator


def completion_with_retry(
    llm: Fireworks,
    use_retry: bool,
    *,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    import fireworks.client

    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @conditional_decorator(use_retry, retry_decorator)
    def _completion_with_retry(**kwargs: Any) -> Any:
        return fireworks.client.Completion.create(
            **kwargs,
        )

    return _completion_with_retry(**kwargs)


async def acompletion_with_retry(
    llm: Fireworks,
    use_retry: bool,
    *,
    run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    import fireworks.client

    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @conditional_decorator(use_retry, retry_decorator)
    async def _completion_with_retry(**kwargs: Any) -> Any:
        return await fireworks.client.Completion.acreate(
            **kwargs,
        )

    return await _completion_with_retry(**kwargs)


def completion_with_retry_batching(
    llm: Fireworks,
    use_retry: bool,
    *,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    import fireworks.client

    prompt = kwargs["prompt"]
    del kwargs["prompt"]

    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @conditional_decorator(use_retry, retry_decorator)
    def _completion_with_retry(prompt: str) -> Any:
        return fireworks.client.Completion.create(**kwargs, prompt=prompt)

    def batch_sync_run() -> List:
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(_completion_with_retry, prompt))
        return results

    return batch_sync_run()


async def acompletion_with_retry_batching(
    llm: Fireworks,
    use_retry: bool,
    *,
    run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    import fireworks.client

    prompt = kwargs["prompt"]
    del kwargs["prompt"]

    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @conditional_decorator(use_retry, retry_decorator)
    async def _completion_with_retry(prompt: str) -> Any:
        return await fireworks.client.Completion.acreate(**kwargs, prompt=prompt)

    def run_coroutine_in_new_loop(
        coroutine_func: Any, *args: Dict, **kwargs: Dict
    ) -> Any:
        new_loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(new_loop)
            return new_loop.run_until_complete(coroutine_func(*args, **kwargs))
        finally:
            new_loop.close()

    async def batch_sync_run() -> List:
        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(
                    run_coroutine_in_new_loop,
                    [_completion_with_retry] * len(prompt),
                    prompt,
                )
            )
        return results

    return await batch_sync_run()


async def acompletion_with_retry_streaming(
    llm: Fireworks,
    use_retry: bool,
    *,
    run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call for streaming."""
    import fireworks.client

    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @conditional_decorator(use_retry, retry_decorator)
    async def _completion_with_retry(**kwargs: Any) -> Any:
        return fireworks.client.Completion.acreate(
            **kwargs,
        )

    return await _completion_with_retry(**kwargs)


def _create_retry_decorator(
    llm: Fireworks,
    *,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:
    """Define retry mechanism."""
    import fireworks.client

    errors = [
        fireworks.client.error.RateLimitError,
        fireworks.client.error.InternalServerError,
        fireworks.client.error.BadGatewayError,
        fireworks.client.error.ServiceUnavailableError,
    ]
    return create_base_retry_decorator(
        error_types=errors, max_retries=llm.max_retries, run_manager=run_manager
    )
