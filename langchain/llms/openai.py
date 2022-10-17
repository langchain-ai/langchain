"""Wrapper around OpenAI APIs."""
import os
from typing import Any, Dict, List, Mapping, Optional

from pydantic import BaseModel, Extra, root_validator

from langchain.llms.base import LLM


class OpenAI(BaseModel, LLM):
    """Wrapper around OpenAI large language models."""

    client: Any
    model_name: str = "text-davinci-002"
    temperature: float = 0.7
    max_tokens: int = 256
    top_p: int = 1
    frequency_penalty: int = 0
    presence_penalty: int = 0
    n: int = 1
    best_of: int = 1

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key python package exists in environment."""
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError(
                "Did not find OpenAI API key, please add an environment variable"
                " `OPENAI_API_KEY` which contains it."
            )
        try:
            import openai

            values["client"] = openai.Completion
        except ImportError:
            raise ValueError(
                "Could not import openai python package. "
                "Please it install it with `pip install openai`."
            )
        return values

    @property
    def default_params(self) -> Mapping[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "n": self.n,
            "best_of": self.best_of,
        }

    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call out to OpenAI's create endpoint."""
        response = self.client.create(
            model=self.model_name, prompt=prompt, stop=stop, **self.default_params
        )
        return response["choices"][0]["text"]
