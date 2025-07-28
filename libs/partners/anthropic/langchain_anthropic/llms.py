from __future__ import annotations

import re
import warnings
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any, Callable, Optional

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
from langchain_core.utils import get_pydantic_field_names
from langchain_core.utils.utils import _build_model_kwargs, from_env, secret_from_env
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self


class _AnthropicCommon(BaseLanguageModel):
    client: Any = None  #: :meta private:
    async_client: Any = None  #: :meta private:
    model: str = Field(default="claude-3-5-sonnet-latest", alias="model_name")
    """Model name to use."""

    max_tokens: int = Field(default=1024, alias="max_tokens_to_sample")
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
    ``ANTHROPIC_API_URL``. If not set, the default value ``https://api.anthropic.com``
    will be used.
    """

    anthropic_api_key: SecretStr = Field(
        alias="api_key",
        default_factory=secret_from_env("ANTHROPIC_API_KEY", default=""),
    )
    """Automatically read from env var ``ANTHROPIC_API_KEY`` if not provided."""

    HUMAN_PROMPT: Optional[str] = None
    AI_PROMPT: Optional[str] = None
    count_tokens: Optional[Callable[[str], int]] = None
    model_kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict) -> Any:
        all_required_field_names = get_pydantic_field_names(cls)
        return _build_model_kwargs(values, all_required_field_names)

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
        # Keep for backward compatibility but not used in Messages API
        self.HUMAN_PROMPT = getattr(anthropic, "HUMAN_PROMPT", None)
        self.AI_PROMPT = getattr(anthropic, "AI_PROMPT", None)
        return self

    @property
    def _default_params(self) -> Mapping[str, Any]:
        """Get the default parameters for calling Anthropic API."""
        d = {
            "max_tokens": self.max_tokens,
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
        return {**self._default_params}

    def _get_anthropic_stop(self, stop: Optional[list[str]] = None) -> list[str]:
        if stop is None:
            stop = []
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
    def raise_warning(cls, values: dict) -> Any:
        """Raise warning that this class is deprecated."""
        warnings.warn(
            "This Anthropic LLM is deprecated. "
            "Please use `from langchain_anthropic import ChatAnthropic` "
            "instead",
            stacklevel=2,
        )
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "anthropic-llm"

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"anthropic_api_key": "ANTHROPIC_API_KEY"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "model_kwargs": self.model_kwargs,
            "streaming": self.streaming,
            "default_request_timeout": self.default_request_timeout,
            "max_retries": self.max_retries,
        }

    def _get_ls_params(
        self,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = super()._get_ls_params(stop=stop, **kwargs)
        identifying_params = self._identifying_params
        if max_tokens := kwargs.get(
            "max_tokens",
            identifying_params.get("max_tokens"),
        ):
            params["ls_max_tokens"] = max_tokens
        return params

    def _format_messages(self, prompt: str) -> list[dict[str, str]]:
        """Convert prompt to Messages API format."""
        messages = []

        # Handle legacy prompts that might have HUMAN_PROMPT/AI_PROMPT markers
        if self.HUMAN_PROMPT and self.HUMAN_PROMPT in prompt:
            # Split on human/assistant turns
            parts = prompt.split(self.HUMAN_PROMPT)

            for _, part in enumerate(parts):
                if not part.strip():
                    continue

                if self.AI_PROMPT and self.AI_PROMPT in part:
                    # Split human and assistant parts
                    human_part, assistant_part = part.split(self.AI_PROMPT, 1)
                    if human_part.strip():
                        messages.append({"role": "user", "content": human_part.strip()})
                    if assistant_part.strip():
                        messages.append(
                            {"role": "assistant", "content": assistant_part.strip()}
                        )
                else:
                    # Just human content
                    if part.strip():
                        messages.append({"role": "user", "content": part.strip()})
        else:
            # Handle modern format or plain text
            # Clean prompt for Messages API
            content = re.sub(r"^\n*Human:\s*", "", prompt)
            content = re.sub(r"\n*Assistant:\s*.*$", "", content)
            if content.strip():
                messages.append({"role": "user", "content": content.strip()})

        # Ensure we have at least one message
        if not messages:
            messages = [{"role": "user", "content": prompt.strip() or "Hello"}]

        return messages

    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        r"""Call out to Anthropic's completion endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.
            run_manager: Optional callback manager for LLM run.
            kwargs: Additional keyword arguments to pass to the model.

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
                prompt=prompt,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            ):
                completion += chunk.text
            return completion

        stop = self._get_anthropic_stop(stop)
        params = {**self._default_params, **kwargs}

        # Remove parameters not supported by Messages API
        params = {k: v for k, v in params.items() if k != "max_tokens_to_sample"}

        response = self.client.messages.create(
            messages=self._format_messages(prompt),
            stop_sequences=stop if stop else None,
            **params,
        )
        return response.content[0].text

    def convert_prompt(self, prompt: PromptValue) -> str:
        return prompt.to_string()

    async def _acall(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to Anthropic's completion endpoint asynchronously."""
        if self.streaming:
            completion = ""
            async for chunk in self._astream(
                prompt=prompt,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            ):
                completion += chunk.text
            return completion

        stop = self._get_anthropic_stop(stop)
        params = {**self._default_params, **kwargs}

        # Remove parameters not supported by Messages API
        params = {k: v for k, v in params.items() if k != "max_tokens_to_sample"}

        response = await self.async_client.messages.create(
            messages=self._format_messages(prompt),
            stop_sequences=stop if stop else None,
            **params,
        )
        return response.content[0].text

    def _stream(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        r"""Call Anthropic completion_stream and return the resulting generator.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.
            run_manager: Optional callback manager for LLM run.
            kwargs: Additional keyword arguments to pass to the model.

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

        # Remove parameters not supported by Messages API
        params = {k: v for k, v in params.items() if k != "max_tokens_to_sample"}

        with self.client.messages.stream(
            messages=self._format_messages(prompt),
            stop_sequences=stop if stop else None,
            **params,
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                    chunk = GenerationChunk(text=event.delta.text)
                    if run_manager:
                        run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                    yield chunk

    async def _astream(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        r"""Call Anthropic completion_stream and return the resulting generator.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.
            run_manager: Optional callback manager for LLM run.
            kwargs: Additional keyword arguments to pass to the model.

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

        # Remove parameters not supported by Messages API
        params = {k: v for k, v in params.items() if k != "max_tokens_to_sample"}

        async with self.async_client.messages.stream(
            messages=self._format_messages(prompt),
            stop_sequences=stop if stop else None,
            **params,
        ) as stream:
            async for event in stream:
                if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                    chunk = GenerationChunk(text=event.delta.text)
                    if run_manager:
                        await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                    yield chunk

    def get_num_tokens(self, text: str) -> int:
        """Calculate number of tokens."""
        msg = (
            "Anthropic's legacy count_tokens method was removed in anthropic 0.39.0 "
            "and langchain-anthropic 0.3.0. Please use "
            "ChatAnthropic.get_num_tokens_from_messages instead."
        )
        raise NotImplementedError(
            msg,
        )


@deprecated(since="0.1.0", removal="1.0.0", alternative="AnthropicLLM")
class Anthropic(AnthropicLLM):
    """Anthropic large language model."""
