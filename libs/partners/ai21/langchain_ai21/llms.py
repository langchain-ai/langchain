import asyncio
from functools import partial
from typing import (
    Any,
    List,
    Mapping,
    Optional,
)

from ai21.models import CompletionsResponse
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import Generation, LLMResult

from langchain_ai21.ai21_base import AI21Base


class AI21LLM(BaseLLM, AI21Base):
    """AI21 large language models. Different model types support different parameters
    and different parameter values. Please read the [AI21 reference documentation]
    (https://docs.ai21.com/reference) for your model to understand which parameters
    are available.

    AI21LLM supports only the older Jurassic models.
    We recommend using ChatAI21 with the newest models, for better results and more
    features.

    Example:
        .. code-block:: python

            from langchain_ai21 import AI21LLM

            model = AI21LLM(
               # defaults to os.environ.get("AI21_API_KEY")
                api_key="my_api_key"
            )
    """

    model: str
    """Model type you wish to interact with. 
    You can view the options at https://github.com/AI21Labs/ai21-python?tab=readme-ov-file#model-types"""

    num_results: int = 1
    """The number of responses to generate for a given prompt."""

    max_tokens: int = 16
    """The maximum number of tokens to generate for each response."""

    min_tokens: int = 0
    """The minimum number of tokens to generate for each response.
    _Not supported for all models._"""

    temperature: float = 0.7
    """A value controlling the "creativity" of the model's responses."""

    top_p: float = 1
    """A value controlling the diversity of the model's responses."""

    top_k_return: int = 0
    """The number of top-scoring tokens to consider for each generation step.
    _Not supported for all models._"""

    frequency_penalty: Optional[Any] = None
    """A penalty applied to tokens that are frequently generated.
    _Not supported for all models._"""

    presence_penalty: Optional[Any] = None
    """ A penalty applied to tokens that are already present in the prompt.
    _Not supported for all models._"""

    count_penalty: Optional[Any] = None
    """A penalty applied to tokens based on their frequency 
    in the generated responses. _Not supported for all models._"""

    custom_model: Optional[str] = None
    epoch: Optional[int] = None

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "ai21-llm"

    @property
    def _default_params(self) -> Mapping[str, Any]:
        base_params = {
            "model": self.model,
            "num_results": self.num_results,
            "max_tokens": self.max_tokens,
            "min_tokens": self.min_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k_return": self.top_k_return,
        }

        if self.count_penalty is not None:
            base_params["count_penalty"] = self.count_penalty.to_dict()

        if self.custom_model is not None:
            base_params["custom_model"] = self.custom_model

        if self.epoch is not None:
            base_params["epoch"] = self.epoch

        if self.frequency_penalty is not None:
            base_params["frequency_penalty"] = self.frequency_penalty.to_dict()

        if self.presence_penalty is not None:
            base_params["presence_penalty"] = self.presence_penalty.to_dict()

        return base_params

    def _build_params_for_request(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> Mapping[str, Any]:
        params = {}

        if stop is not None:
            if "stop" in kwargs:
                raise ValueError("stop is defined in both stop and kwargs")
            params["stop_sequences"] = stop

        return {
            **self._default_params,
            **params,
            **kwargs,
        }

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations: List[List[Generation]] = []
        token_count = 0

        params = self._build_params_for_request(stop=stop, **kwargs)

        for prompt in prompts:
            response = self._invoke_completion(prompt=prompt, **params)
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
        **kwargs: Any,
    ) -> CompletionsResponse:
        return self.client.completion.create(
            prompt=prompt,
            **kwargs,
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
