"""Wrapper around Anthropic APIs."""
import re
from typing import Any, Dict, Generator, List, Mapping, Optional

from pydantic import BaseModel, Extra, root_validator

from langchain.schema import EnvAuthStrategy
from langchain.llms.base import LLM
from langchain.utils import get_from_dict_or_env

class AnthropicAuthStrategy(EnvAuthStrategy):
    name = "ANTHROPIC_API_KEY"

class Anthropic(LLM, BaseModel):
    r"""Wrapper around Anthropic large language models.

    To use, you should have the ``anthropic`` python package installed, and the
    environment variable ``ANTHROPIC_API_KEY`` set with your API key, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python
            import anthropic
            from langchain.llms import Anthropic
            model = Anthropic(model="<model_name>", anthropic_api_key="my-api-key")

            # Simplest invocation, automatically wrapped with HUMAN_PROMPT
            # and AI_PROMPT.
            response = model("What are the biggest risks facing humanity?")

            # Or if you want to use the chat mode, build a few-shot-prompt, or
            # put words in the Assistant's mouth, use HUMAN_PROMPT and AI_PROMPT:
            raw_prompt = "What are the biggest risks facing humanity?"
            prompt = f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}"
            response = model(prompt)
    """

    id = "anthropic"
    """Unique ID for this provider class."""

    model_id = "claude-v1"
    """Model ID to invoke by this provider via generate/agenerate."""

    # Anthropic model provider supports any model available via
    # `anthropic.Client#completion()`.
    # Reference: https://console.anthropic.com/docs/api/reference
    models = ["claude-v1", "claude-v1.0", "claude-v1.2", "claude-instant-v1", "claude-instant-v1.0"]
    """List of supported models by their IDs. For registry providers, this will
    be just ["*"]."""

    pypi_package_deps = ["anthropic"]
    """List of PyPi package dependencies."""

    auth_strategy = AnthropicAuthStrategy
    """Authentication/authorization strategy. Declares what credentials are
    required to use this model provider. Generally should not be `None`."""

    client: Any  #: :meta private:

    max_tokens_to_sample: int = 256
    """Denotes the number of tokens to predict per generation."""

    temperature: float = 1.0
    """A non-negative float that tunes the degree of randomness in generation."""

    top_k: int = 0
    """Number of most likely tokens to consider at each step."""

    top_p: float = 1
    """Total probability mass of tokens to consider at each step."""

    anthropic_api_key: Optional[str] = None

    HUMAN_PROMPT: Optional[str] = None
    AI_PROMPT: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        anthropic_api_key = get_from_dict_or_env(
            values, "anthropic_api_key", "ANTHROPIC_API_KEY"
        )
        try:
            import anthropic

            values["client"] = anthropic.Client(anthropic_api_key)
            values["HUMAN_PROMPT"] = anthropic.HUMAN_PROMPT
            values["AI_PROMPT"] = anthropic.AI_PROMPT
        except ImportError:
            raise ValueError(
                "Could not import anthropic python package. "
                "Please it install it with `pip install anthropic`."
            )
        return values

    @property
    def _default_params(self) -> Mapping[str, Any]:
        """Get the default parameters for calling Anthropic API."""
        return {
            "max_tokens_to_sample": self.max_tokens_to_sample,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_id": self.model_id}, **self._default_params}

    def _wrap_prompt(self, prompt: str) -> str:
        if not self.HUMAN_PROMPT or not self.AI_PROMPT:
            raise NameError("Please ensure the anthropic package is loaded")

        if prompt.startswith(self.HUMAN_PROMPT):
            return prompt  # Already wrapped.

        # Guard against common errors in specifying wrong number of newlines.
        corrected_prompt, n_subs = re.subn(r"^\n*Human:", self.HUMAN_PROMPT, prompt)
        if n_subs == 1:
            return corrected_prompt

        # As a last resort, wrap the prompt ourselves to emulate instruct-style.
        return f"{self.HUMAN_PROMPT} {prompt}{self.AI_PROMPT} Sure, here you go:\n"

    def _get_anthropic_stop(self, stop: Optional[List[str]] = None) -> List[str]:
        if not self.HUMAN_PROMPT or not self.AI_PROMPT:
            raise NameError("Please ensure the anthropic package is loaded")

        if stop is None:
            stop = []

        # Never want model to invent new turns of Human / Assistant dialog.
        stop.extend([self.HUMAN_PROMPT, self.AI_PROMPT])

        return stop

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        r"""Call out to Anthropic's completion endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                prompt = "What are the biggest risks facing humanity?"
                prompt = f"\n\nHuman: {prompt}\n\nAssistant:"
                response = model(prompt)

        """
        stop = self._get_anthropic_stop(stop)
        response = self.client.completion(
            model=self.model_id,
            prompt=self._wrap_prompt(prompt),
            stop_sequences=stop,
            **self._default_params,
        )
        text = response["completion"]
        return text

    def stream(self, prompt: str, stop: Optional[List[str]] = None) -> Generator:
        r"""Call Anthropic completion_stream and return the resulting generator.

        BETA: this is a beta feature while we figure out the right abstraction.
        Once that happens, this interface could change.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            A generator representing the stream of tokens from Anthropic.

        Example:
            .. code-block:: python


                prompt = "Write a poem about a stream."
                prompt = f"\n\nHuman: {prompt}\n\nAssistant:"
                generator = anthropic.stream(prompt)
                for token in generator:
                    yield token
        """
        stop = self._get_anthropic_stop(stop)
        return self.client.completion_stream(
            model=self.model_id,
            prompt=self._wrap_prompt(prompt),
            stop_sequences=stop,
            **self._default_params,
        )
