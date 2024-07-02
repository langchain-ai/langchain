"""NexaAI large language models."""

import logging
from os import getenv
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import requests
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import Generation, LLMResult
from pydantic import Field

logger = logging.getLogger(__name__)


class NexaAILLM(BaseLLM):
    """NexaAILLM large language models.

    Example:
        .. code-block:: python

            from langchain_nexa_ai import NexaAILLM

            model = NexaAILLM()
            model.invoke("Come up with 10 names for a song about parrots")
    """

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "nexa-ai-llm"

    api_key: Optional[str]
    url: str = "https://octopus.nexa4ai.com/model/octopus-v2"
    response_keys: Tuple = ("result", "function_name", "function_arguments", "latency")
    headers: Dict[str, str] = Field(default_factory=dict)

    def __init__(self, api_key: Optional[str] = None, **kwargs: Any):
        """Initialize NexaAILLM.

        Args:
            api_key: API key.
        """
        super().__init__(**kwargs)
        self._init_request_template(api_key)

    def _init_request_template(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or getenv("NEXA_API_KEY")
        if self.api_key is None:
            raise ValueError(
                "NEXA_API_KEY must be set in the environment or "
                "passed via argument `api_key`."
            )

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _create_llm_result(
        self,
        prompts: List[str],
        categories: List[str],
    ) -> LLMResult:
        generations: List[List[Generation]] = []
        llm_outputs: Dict[str, List[str]] = {}

        for p, c in zip(prompts, categories):
            data = {
                "input_text": p,
                "category": c,
            }
            response = requests.post(self.url, headers=self.headers, json=data)
            response_json = response.json()

            generations.append(
                [
                    Generation(
                        text=response.text,
                        generation_info=response_json,
                    )
                ]
            )
            for k in self.response_keys:
                llm_outputs[k] = llm_outputs.get(k, []) + [response_json.get(k, None)]

        return LLMResult(generations=generations, llm_output=llm_outputs)

    def _prepare_categories(self, prompts: List[str], **kwargs: Any) -> List[str]:
        categories: List = kwargs.get("categories", None)
        category: str = kwargs.get("category", None)

        if categories is None:
            if category is None:
                logger.warning(
                    "No category or categories provided. Defaulting to 'shopping'."
                )
                category = "shopping"
            categories = [
                category,
            ] * len(prompts)
        elif category is not None:
            logger.warning("Both category and categories provided. Using categories.")

        assert len(categories) == len(
            prompts
        ), "Number of prompts and categories must match."
        return categories

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        categories = self._prepare_categories(prompts, **kwargs)
        return self._create_llm_result(prompts, categories)
