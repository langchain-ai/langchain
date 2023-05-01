"""Wrapper arround Google's PaLM Text APIs."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, root_validator

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.llms import BaseLLM
from langchain.schema import Generation, LLMResult
from langchain.utils import get_from_dict_or_env


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


class GooglePalm(BaseLLM, BaseModel):
    client: Any  #: :meta private:
    google_api_key: Optional[str]
    model_name: str = "models/text-bison-001"
    """Model name to use."""
    temperature: float = 0.7
    """Run inference with this temperature. Must by in the closed interval
       [0.0, 1.0]."""
    top_p: Optional[float] = None
    """Decode using nucleus sampling: consider the smallest set of tokens whose
       probability sum is at least top_p. Must be in the closed interval [0.0, 1.0]."""
    top_k: Optional[int] = None
    """Decode using top-k sampling: consider the set of top_k most probable tokens.
       Must be positive."""
    max_output_tokens: Optional[int] = None
    """Maximum number of tokens to include in a candidate. Must be greater than zero.
       If unset, will default to 64."""
    n: int = 1
    """Number of chat completions to generate for each prompt. Note that the API may
       not return the full n completions if duplicates are generated."""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate api key, python package exists."""
        google_api_key = get_from_dict_or_env(
            values, "google_api_key", "GOOGLE_API_KEY"
        )
        try:
            import google.generativeai as genai

            genai.configure(api_key=google_api_key)
        except ImportError:
            raise ImportError("Could not import google.generativeai python package.")

        values["client"] = genai

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
            completion = self.client.generate_text(
                model=self.model_name,
                prompt=prompt,
                stop_sequences=stop,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_output_tokens=self.max_output_tokens,
                candidate_count=self.n,
            )

            prompt_generations = []
            for candidate in completion.candidates:
                raw_text = candidate["output"]
                stripped_text = _strip_erroneous_leading_spaces(raw_text)
                prompt_generations.append(Generation(text=stripped_text))
            generations.append(prompt_generations)

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
        return "google_palm"
