"""Wrapper around Anthropic APIs."""
import re
import warnings
from typing import Any, Callable, Dict, Generator, List, Mapping, Optional, Tuple, Union

from pydantic import BaseModel, Extra, root_validator

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.llms.base import LLM
from langchain.utils import get_from_dict_or_env


class _AnthropicCommon(BaseModel):
    client: Any = None  #: :meta private:
    model: str = "claude-v1"
    """Model name to use."""

    max_tokens_to_sample: int = 256
    """Denotes the number of tokens to predict per generation."""

    temperature: Optional[float] = None
    """A non-negative float that tunes the degree of randomness in generation."""

    top_k: Optional[int] = None
    """Number of most likely tokens to consider at each step."""

    top_p: Optional[float] = None
    """Total probability mass of tokens to consider at each step."""

    streaming: bool = False
    """Whether to stream the results."""

    default_request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    """Timeout for requests to Anthropic Completion API. Default is 600 seconds."""

    anthropic_api_key: Optional[str] = None

    HUMAN_PROMPT: Optional[str] = None
    AI_PROMPT: Optional[str] = None
    count_tokens: Optional[Callable[[str], int]] = None

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        anthropic_api_key = get_from_dict_or_env(
            values, "anthropic_api_key", "ANTHROPIC_API_KEY"
        )
        try:
            import anthropic

            values["client"] = anthropic.Client(
                api_key=anthropic_api_key,
                default_request_timeout=values["default_request_timeout"],
            )
            values["HUMAN_PROMPT"] = anthropic.HUMAN_PROMPT
            values["AI_PROMPT"] = anthropic.AI_PROMPT
            values["count_tokens"] = anthropic.count_tokens
        except ImportError:
            raise ValueError(
                "Could not import anthropic python package. "
                "Please it install it with `pip install anthropic`."
            )
        return values

    @property
    def _default_params(self) -> Mapping[str, Any]:
        """Get the default parameters for calling Anthropic API."""
        d = {
            "max_tokens_to_sample": self.max_tokens_to_sample,
            "model": self.model,
        }
        if self.temperature is not None:
            d["temperature"] = self.temperature
        if self.top_k is not None:
            d["top_k"] = self.top_k
        if self.top_p is not None:
            d["top_p"] = self.top_p
        return d

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{}, **self._default_params}

    def _get_anthropic_stop(self, stop: Optional[List[str]] = None) -> List[str]:
        if not self.HUMAN_PROMPT or not self.AI_PROMPT:
            raise NameError("Please ensure the anthropic package is loaded")

        if stop is None:
            stop = []

        # Never want model to invent new turns of Human / Assistant dialog.
        stop.extend([self.HUMAN_PROMPT])

        return stop

    def get_num_tokens(self, text: str) -> int:
        """Calculate number of tokens."""
        if not self.count_tokens:
            raise NameError("Please ensure the anthropic package is loaded")
        return self.count_tokens(text)


class Anthropic(LLM, _AnthropicCommon):
    r"""Wrapper around Anthropic's large language models.

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

    @root_validator()
    def raise_warning(cls, values: Dict) -> Dict:
        """Raise warning that this class is deprecated."""
        warnings.warn(
            "This Anthropic LLM is deprecated. "
            "Please use `from langchain.chat_models import ChatAnthropic` instead"
        )
        return values

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "anthropic-llm"

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

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
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
        if self.streaming:
            stream_resp = self.client.completion_stream(
                prompt=self._wrap_prompt(prompt),
                stop_sequences=stop,
                **self._default_params,
            )
            current_completion = ""
            for data in stream_resp:
                delta = data["completion"][len(current_completion) :]
                current_completion = data["completion"]
                if run_manager:
                    run_manager.on_llm_new_token(delta, **data)
            return current_completion
        response = self.client.completion(
            prompt=self._wrap_prompt(prompt),
            stop_sequences=stop,
            **self._default_params,
        )
        return response["completion"]

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> str:
        """Call out to Anthropic's completion endpoint asynchronously."""
        stop = self._get_anthropic_stop(stop)
        if self.streaming:
            stream_resp = await self.client.acompletion_stream(
                prompt=self._wrap_prompt(prompt),
                stop_sequences=stop,
                **self._default_params,
            )
            current_completion = ""
            async for data in stream_resp:
                delta = data["completion"][len(current_completion) :]
                current_completion = data["completion"]
                if run_manager:
                    await run_manager.on_llm_new_token(delta, **data)
            return current_completion
        response = await self.client.acompletion(
            prompt=self._wrap_prompt(prompt),
            stop_sequences=stop,
            **self._default_params,
        )
        return response["completion"]

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
            prompt=self._wrap_prompt(prompt),
            stop_sequences=stop,
            **self._default_params,
        )
