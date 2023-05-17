"""Wrapper arround Google Cloud Platform Vertex AI PaLM Text APIs."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, root_validator
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.llms import BaseLLM
from langchain.schema import Generation, LLMResult

logger = logging.getLogger(__name__)


def _strip_erroneous_leading_spaces(text: str) -> str:
    """Strip erroneous leading spaces from text.

    The PaLM API will sometimes erroneously return a single leading space in all
    lines > 1. This function strips that space.
    """
    has_leading_space = all(not line or line[0] == " " for line in text.split("\n")[1:])
    if has_leading_space:
        return text.replace("\n ", "\n")
    else:
        return text


class GoogleCloudVertexAIPalm(BaseLLM, BaseModel):
    """Wrapper around Google Cloud's Vertex AI PaLM Text Generation API.

    To use you must have the google-cloud-aiplatform Python package installed and
        either:

        1. Have credentials configured for your environment -
            (gcloud, workload identity, etc...)
        2. Store the path to a service account JSON file
            as the GOOGLE_APPLICATION_CREDENTIALS environment variable

    Example:
        .. code-block:: python

            from langchain.llms import GoogleVertexAIPalm
            llm = VertexAIGooglePalm()

    """

    client: Any  #: :meta private:
    model_name: str = "text-bison@001"
    """Model name to use."""
    temperature: float = 0.2
    """Run inference with this temperature. Must by in the closed interval
       [0.0, 1.0]."""
    top_p: Optional[float] = 0.8
    """Decode using nucleus sampling: consider the smallest set of tokens whose
       probability sum is at least top_p. Must be in the closed interval [0.0, 1.0]."""
    top_k: Optional[int] = 40
    """Decode using top-k sampling: consider the set of top_k most probable tokens.
       Must be positive."""
    max_output_tokens: Optional[int] = 256
    """Maximum number of tokens to include in a candidate. Must be greater than zero.
       If unset, will default to 256."""
    tuned_model: Optional[bool] = False
    """Whether or not the model_name referenced is a fine-tuned model. 
    Defaults to False."""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate auth and that the python package exists."""
        import google.auth

        credentials, project_id = google.auth.default()

        try:
            from vertexai.preview.language_models import TextGenerationModel

        except ImportError:
            raise ImportError(
                "Could not import vertexai python package."
                "Try running `pip install google-cloud-aiplatform>=1.25.0`"
            )

        if values["tuned_model"]:
            values["client"] = TextGenerationModel.get_tuned_model(values["model_name"])
        else:
            values["client"] = TextGenerationModel.from_pretrained(values["model_name"])

        if values["temperature"] is not None and not 0 <= values["temperature"] <= 1:
            raise ValueError("temperature must be in the range [0.0, 1.0]")

        if values["top_p"] is not None and not 0 <= values["top_p"] <= 1:
            raise ValueError("top_p must be in the range [0.0, 1.0]")

        if values["top_k"] is not None and values["top_k"] <= 0:
            raise ValueError("top_k must be positive")

        if values["max_output_tokens"] is not None and values["max_output_tokens"] <= 0:
            raise ValueError("max_output_tokens must be greater than zero")

        return values

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            completion_with_retry = retry(
                reraise=True,
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=4, max=10),
                before_sleep=before_sleep_log(logger, logging.WARNING),
            )(self.client.predict)
            result = completion_with_retry(
                prompt,
                max_output_tokens=self.max_output_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )
            stripped_text = _strip_erroneous_leading_spaces(result.text)
            generations.append([Generation(text=stripped_text)])

        return LLMResult(generations=generations)

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> LLMResult:
        raise NotImplementedError()

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "google_cloud_vertex_ai_palm"
