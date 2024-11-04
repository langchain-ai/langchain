import re
import warnings
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
)

import anthropic
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseLanguageModel, LangSmithParams
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.prompt_values import PromptValue
from langchain_core.utils import (
    get_pydantic_field_names,
)
from langchain_core.utils.utils import (
    _build_model_kwargs,
    from_env,
    secret_from_env,
)
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self


class _AnthropicCommon(BaseLanguageModel):
    client: Any = None  #: :meta private:
    async_client: Any = None  #: :meta private:
    model: str = Field(default="claude-2", alias="model_name")
    """Model name to use."""

    max_tokens_to_sample: int = Field(default=1024, alias="max_tokens")
    """Denotes the number of tokens to predict per generation."""

    temperature: Optional[float] = None
    """A non-negative float that tunes the degree of randomness in generation."""

    top_k: Optional[int] = None
    """Number of most likely tokens to consider at each step."""

    top_p: Optional[float] = None
    """Total probability mass of tokens to consider at each step."""

    streaming: bool = False
    """Whether to stream the results."""

    default_request_timeout: Optional[float] = None
    """Timeout for requests to Anthropic Completion API. Default is 600 seconds."""

    max_retries: int = 2
    """Number of retries allowed for requests sent to the Anthropic Completion API."""

    anthropic_api_url: Optional[str] = Field(
        alias="base_url",
        default_factory=from_env(
            "ANTHROPIC_API_URL",
            default="https://api.anthropic.com",
        ),
    )
    """Base URL for API requests. Only specify if using a proxy or service emulator.

    If a value isn't passed in, will attempt to read the value from
    ANTHROPIC_API_URL. If not set, the default value of 'https://api.anthropic.com' will
    be used.
    """

    anthropic_api_key: SecretStr = Field(
        alias="api_key",
        default_factory=secret_from_env("ANTHROPIC_API_KEY", default=""),
    )
    """Automatically read from env var `ANTHROPIC_API_KEY` if not provided."""

    HUMAN_PROMPT: Optional[str] = None
    AI_PROMPT: Optional[str] = None
    count_tokens: Optional[Callable[[str], int]] = None
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: Dict) -> Any:
        all_required_field_names = get_pydantic_field_names(cls)
        values = _build_model_kwargs(values, all_required_field_names)
        return values

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        self.client = anthropic.Anthropic(
            base_url=self.anthropic_api_url,
            api_key=self.anthropic_api_key.get_secret_value(),
            timeout=self.default_request_timeout,
            max_retries=self.max_retries,
        )
        self.async_client = anthropic.AsyncAnthropic(
            base_url=self.anthropic_api_url,
            api_key=self.anthropic_api_key.get_secret_value(),
            timeout=self.default_request_timeout,
            max_retries=self.max_retries,
        )
        self.HUMAN_PROMPT = anthropic.HUMAN_PROMPT
        self.AI_PROMPT = anthropic.AI_PROMPT
        self.count_tokens = self.client.count_tokens
        return self

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
        return {**d, **self.model_kwargs}

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


class AnthropicLLM(LLM, _AnthropicCommon):
    """Anthropic large language model.

    To use, you should have the environment variable ``ANTHROPIC_API_KEY``
    set with your API key, or pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_anthropic import AnthropicLLM

            model = AnthropicLLM()
    """

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="before")
    @classmethod
    def raise_warning(cls, values: Dict) -> Any:
        """Raise warning that this class is deprecated."""
        warnings.warn(
            "This Anthropic LLM is deprecated. "
            "Please use `from langchain_anthropic import ChatAnthropic` "
            "instead"
        )
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "anthropic-llm"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"anthropic_api_key": "ANTHROPIC_API_KEY"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens_to_sample,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "model_kwargs": self.model_kwargs,
            "streaming": self.streaming,
            "default_request_timeout": self.default_request_timeout,
            "max_retries": self.max_retries,
        }

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = super()._get_ls_params(stop=stop, **kwargs)
        identifying_params = self._identifying_params
        if max_tokens := kwargs.get(
            "max_tokens_to_sample",
            identifying_params.get("max_tokens"),
        ):
            params["ls_max_tokens"] = max_tokens
        return params

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
        **kwargs: Any,
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
                response = model.invoke(prompt)

        """
        if self.streaming:
            completion = ""
            for chunk in self._stream(
                prompt=prompt, stop=stop, run_manager=run_manager, **kwargs
            ):
                completion += chunk.text
            return completion

        stop = self._get_anthropic_stop(stop)
        params = {**self._default_params, **kwargs}
        response = self.client.completions.create(
            prompt=self._wrap_prompt(prompt),
            stop_sequences=stop,
            **params,
        )
        return response.completion

    def convert_prompt(self, prompt: PromptValue) -> str:
        return self._wrap_prompt(prompt.to_string())

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to Anthropic's completion endpoint asynchronously."""
        if self.streaming:
            completion = ""
            async for chunk in self._astream(
                prompt=prompt, stop=stop, run_manager=run_manager, **kwargs
            ):
                completion += chunk.text
            return completion

        stop = self._get_anthropic_stop(stop)
        params = {**self._default_params, **kwargs}

        response = await self.async_client.completions.create(
            prompt=self._wrap_prompt(prompt),
            stop_sequences=stop,
            **params,
        )
        return response.completion

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        r"""Call Anthropic completion_stream and return the resulting generator.

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
        params = {**self._default_params, **kwargs}

        for token in self.client.completions.create(
            prompt=self._wrap_prompt(prompt), stop_sequences=stop, stream=True, **params
        ):
            chunk = GenerationChunk(text=token.completion)

            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        r"""Call Anthropic completion_stream and return the resulting generator.

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
        params = {**self._default_params, **kwargs}

        async for token in await self.async_client.completions.create(
            prompt=self._wrap_prompt(prompt),
            stop_sequences=stop,
            stream=True,
            **params,
        ):
            chunk = GenerationChunk(text=token.completion)

            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    def get_num_tokens(self, text: str) -> int:
        """Calculate number of tokens."""
        if not self.count_tokens:
            raise NameError("Please ensure the anthropic package is loaded")
        return self.count_tokens(text)


@deprecated(since="0.1.0", removal="0.3.0", alternative="AnthropicLLM")
class Anthropic(AnthropicLLM):
    """Anthropic large language model."""

    pass
