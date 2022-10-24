"""Wrapper around Cohere APIs."""
import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, root_validator

from langchain.llms.base import LLM


def remove_stop_tokens(text: str, stop: List[str]) -> str:
    """Remove stop tokens, should they occur at end."""
    for s in stop:
        if text.endswith(s):
            return text[: -len(s)]
    return text


class Cohere(BaseModel, LLM):
    """Wrapper around Cohere large language models."""

    client: Any
    model: str = "gptd-instruct-tft"
    max_tokens: int = 256
    temperature: float = 0.6
    k: int = 0
    p: int = 1
    frequency_penalty: int = 0
    presence_penalty: int = 0

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def template_is_valid(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        if "COHERE_API_KEY" not in os.environ:
            raise ValueError(
                "Did not find Cohere API key, please add an environment variable"
                " `COHERE_API_KEY` which contains it."
            )
        try:
            import cohere

            values["client"] = cohere.Client(os.environ["COHERE_API_KEY"])
        except ImportError:
            raise ValueError(
                "Could not import cohere python package. "
                "Please it install it with `pip install cohere`."
            )
        return values

    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call out to Cohere's generate endpoint."""
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            k=self.k,
            p=self.p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop_sequences=stop,
        )
        text = response.generations[0].text
        # If stop tokens are provided, Cohere's endpoint returns them.
        # In order to make this consistent with other endpoints, we strip them.
        if stop is not None:
            text = remove_stop_tokens(text, stop)
        return text
