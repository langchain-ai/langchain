import asyncio
from functools import partial
from typing import (
    Any,
    AsyncIterator,
    Iterator,
    List,
    Optional,
)

from ai21.models import CompletionsResponse

from langchain_ai21.ai21_base import AI21Base
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import GenerationChunk, LLMResult, Generation, RunInfo


class AI21LLM(BaseLLM, AI21Base):
    """AI21LLM large language models.

    Example:
        .. code-block:: python

            from langchain_ai21 import AI21LLM

            model = AI21LLM()
    """

    model: str = "j2-ultra"

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "ai21-llm"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        for prompt in prompts:
            response = self._invoke_completion(
                prompt=prompt, model=self.model, stop_sequences=stop, **kwargs
            )

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        # Change implementation if integration natively supports async generation.
        return await asyncio.get_running_loop().run_in_executor(
            None, partial(self._generate, **kwargs), prompts, stop, run_manager
        )

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        raise NotImplementedError

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        yield GenerationChunk(text="Yield chunks")
        yield GenerationChunk(text=" like this!")

    def _invoke_completion(
        self,
        prompt: str,
        model: str,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> CompletionsResponse:
        return self.client.completion.create(
            prompt=prompt,
            model=model,
            max_tokens=kwargs.get("max_tokens"),
            num_results=kwargs.get("num_results"),
            min_tokens=kwargs.get("min_tokens"),
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            top_k_return=kwargs.get("top_k_returns"),
            custom_model=kwargs.get("custom_model"),
            stop_sequences=stop_sequences,
            frequency_penalty=kwargs.get("frequency_penalty"),
            presence_penalty=kwargs.get("presence_penalty"),
            count_penalty=kwargs.get("count_penalty"),
            epoch=kwargs.get("epoch"),
        )

    def _response_to_llm_result(self, response: CompletionsResponse):
        generations = [
            Generation(
                text=completion.data.text,
                generation_info=completion.to_dict(),
            )
            for completion in response.completions
        ]
        return LLMResult(
            generations=[generations],
            run=[RunInfo(run_id=response.id)],
        )
