import asyncio
from functools import partial
from typing import (
    Any,
    List,
    Optional,
)

from ai21.models import CompletionsResponse, Penalty
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import Generation, LLMResult

from langchain_ai21.ai21_base import AI21Base


class AI21LLM(BaseLLM, AI21Base):
    """AI21LLM large language models.

    Example:
        .. code-block:: python

            from langchain_ai21 import AI21LLM

            model = AI21LLM()
    """

    model: str
    """Model type you wish to interact with. 
    You can view the options at https://github.com/AI21Labs/ai21-python?tab=readme-ov-file#model-types"""

    num_results: int = 1
    """The number of responses to generate for a given prompt."""

    max_tokens: int = 16
    """The maximum number of tokens to generate for each response."""

    min_tokens: int = 0
    """The minimum number of tokens to generate for each response."""

    temperature: float = 0.7
    """A value controlling the "creativity" of the model's responses."""

    top_p: float = 1
    """A value controlling the diversity of the model's responses."""

    top_k_returns: int = 0
    """The number of top-scoring tokens to consider for each generation step."""

    frequency_penalty: Optional[Penalty] = None
    """A penalty applied to tokens that are frequently generated."""

    presence_penalty: Optional[Penalty] = None
    """ A penalty applied to tokens that are already present in the prompt."""

    count_penalty: Optional[Penalty] = None
    """A penalty applied to tokens based on their frequency 
    in the generated responses."""

    custom_model: Optional[str] = None
    epoch: Optional[int] = None

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

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
        generations: List[List[Generation]] = []
        token_count = 0

        for prompt in prompts:
            response = self._invoke_completion(
                prompt=prompt, model=self.model, stop_sequences=stop, **kwargs
            )
            generation = self._response_to_generation(response)
            generations.append(generation)
            token_count += self.client.count_tokens(prompt)

        llm_output = {"token_count": token_count, "model_name": self.model}
        return LLMResult(generations=generations, llm_output=llm_output)

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

    def _invoke_completion(
        self,
        prompt: str,
        model: str,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> CompletionsResponse:
        return self.client.completion.create(
            prompt=prompt,
            model=model,
            max_tokens=self.max_tokens,
            num_results=self.num_results,
            min_tokens=self.min_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k_return=self.top_k_returns,
            custom_model=self.custom_model,
            stop_sequences=stop_sequences,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            count_penalty=self.count_penalty,
            epoch=self.epoch,
        )

    def _response_to_generation(
        self, response: CompletionsResponse
    ) -> List[Generation]:
        return [
            Generation(
                text=completion.data.text,
                generation_info=completion.to_dict(),
            )
            for completion in response.completions
        ]
