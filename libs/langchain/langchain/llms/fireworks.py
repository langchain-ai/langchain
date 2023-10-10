from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Union

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.llms.base import LLM, create_base_retry_decorator
from langchain.pydantic_v1 import Field, root_validator
from langchain.schema.language_model import LanguageModelInput
from langchain.schema.output import GenerationChunk
from langchain.schema.runnable.config import RunnableConfig
from langchain.utils.env import get_from_dict_or_env


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


class Fireworks(LLM):
    """Fireworks models."""

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

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key in environment."""
        try:
            import fireworks.client
        except ImportError as e:
            raise ImportError("") from e
        fireworks_api_key = get_from_dict_or_env(
            values, "fireworks_api_key", "FIREWORKS_API_KEY"
        )
        fireworks.client.api_key = fireworks_api_key
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fireworks"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given prompt and input."""
        params: dict = {
            "model": self.model,
            "prompt": prompt,
            **self.model_kwargs,
        }
        response = completion_with_retry(
            self, run_manager=run_manager, stop=stop, **params
        )

        return response.choices[0].text

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given prompt and input."""
        params = {
            "model": self.model,
            "prompt": prompt,
            **self.model_kwargs,
        }
        response = await acompletion_with_retry(
            self, run_manager=run_manager, stop=stop, **params
        )

        return response.choices[0].text

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
            self, run_manager=run_manager, stop=stop, **params
        ):
            chunk = _stream_response_to_generation_chunk(stream_resp)
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
            self, run_manager=run_manager, stop=stop, **params
        ):
            chunk = _stream_response_to_generation_chunk(stream_resp)
            yield chunk

    def stream(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        prompt = self._convert_input(input).to_string()
        generation: Optional[GenerationChunk] = None
        for chunk in self._stream(prompt):
            yield chunk.text
            if generation is None:
                generation = chunk
            else:
                generation += chunk
        assert generation is not None

    async def astream(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        prompt = self._convert_input(input).to_string()
        generation: Optional[GenerationChunk] = None
        async for chunk in self._astream(prompt):
            yield chunk.text
            if generation is None:
                generation = chunk
            else:
                generation += chunk
        assert generation is not None


def completion_with_retry(
    llm: Fireworks,
    *,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    import fireworks.client

    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    def _completion_with_retry(**kwargs: Any) -> Any:
        return fireworks.client.Completion.create(
            **kwargs,
        )

    return _completion_with_retry(**kwargs)


async def acompletion_with_retry(
    llm: Fireworks,
    *,
    run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    import fireworks.client

    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        return await fireworks.client.Completion.acreate(
            **kwargs,
        )

    return await _completion_with_retry(**kwargs)


async def acompletion_with_retry_streaming(
    llm: Fireworks,
    *,
    run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call for streaming."""
    import fireworks.client

    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
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
        fireworks.client.error.ServiceUnavailableError,
    ]
    return create_base_retry_decorator(
        error_types=errors, max_retries=llm.max_retries, run_manager=run_manager
    )
