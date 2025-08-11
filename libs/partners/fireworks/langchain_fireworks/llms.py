"""Wrapper around Fireworks AI's Completion API."""

from __future__ import annotations

import logging
from typing import Any, Optional

import requests
from aiohttp import ClientSession, ClientTimeout
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import LLM
from langchain_core.utils import get_pydantic_field_names
from langchain_core.utils.utils import _build_model_kwargs, secret_from_env
from pydantic import ConfigDict, Field, SecretStr, model_validator

from langchain_fireworks.version import __version__

logger = logging.getLogger(__name__)


class Fireworks(LLM):
    """LLM models from `Fireworks`.

    To use, you'll need an `API key <https://fireworks.ai>`__. This can be passed in as
    init param ``fireworks_api_key`` or set as environment variable
    ``FIREWORKS_API_KEY``.

    `Fireworks AI API reference <https://readme.fireworks.ai/>`__

    Example:

        .. code-block:: python
            response = fireworks.generate(["Tell me a joke."])

    """

    base_url: str = "https://api.fireworks.ai/inference/v1/completions"
    """Base inference API URL."""
    fireworks_api_key: SecretStr = Field(
        alias="api_key",
        default_factory=secret_from_env(
            "FIREWORKS_API_KEY",
            error_message=(
                "You must specify an api key. "
                "You can pass it an argument as `api_key=...` or "
                "set the environment variable `FIREWORKS_API_KEY`."
            ),
        ),
    )
    """Fireworks API key.

    Automatically read from env variable ``FIREWORKS_API_KEY`` if not provided.
    """
    model: str
    """Model name. `(Available models) <https://readme.fireworks.ai/>`__"""
    temperature: Optional[float] = None
    """Model temperature."""
    top_p: Optional[float] = None
    """Used to dynamically adjust the number of choices for each predicted token based
    on the cumulative probabilities. A value of ``1`` will always yield the same output.
    A temperature less than ``1`` favors more correctness and is appropriate for
    question answering or summarization. A value greater than ``1`` introduces more
    randomness in the output.
    """
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for ``create`` call not explicitly specified."""
    top_k: Optional[int] = None
    """Used to limit the number of choices for the next predicted word or token. It
    specifies the maximum number of tokens to consider at each step, based on their
    probability of occurrence. This technique helps to speed up the generation process
    and can improve the quality of the generated text by focusing on the most likely
    options.
    """
    max_tokens: Optional[int] = None
    """The maximum number of tokens to generate."""
    repetition_penalty: Optional[float] = None
    """A number that controls the diversity of generated text by reducing the likelihood
    of repeated sequences. Higher values decrease repetition.
    """
    logprobs: Optional[int] = None
    """An integer that specifies how many top token log probabilities are included in
    the response for each token generation step.
    """
    timeout: Optional[int] = 30
    """Timeout in seconds for requests to the Fireworks API."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        return _build_model_kwargs(values, all_required_field_names)

    @property
    def _llm_type(self) -> str:
        """Return type of model."""
        return "fireworks"

    def _format_output(self, output: dict) -> str:
        return output["choices"][0]["text"]

    @staticmethod
    def get_user_agent() -> str:
        return f"langchain-fireworks/{__version__}"

    @property
    def default_params(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to Fireworks's text generation endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop sequences to use.
            run_manager: (Not used) Optional callback manager for LLM run.
            kwargs: Additional parameters to pass to the model.

        Returns:
            The string generated by the model.

        """
        headers = {
            "Authorization": f"Bearer {self.fireworks_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        stop_to_use = stop[0] if stop and len(stop) == 1 else stop
        payload: dict[str, Any] = {
            **self.default_params,
            "prompt": prompt,
            "stop": stop_to_use,
            **kwargs,
        }

        # filter None values to not pass them to the http payload
        payload = {k: v for k, v in payload.items() if v is not None}
        response = requests.post(
            url=self.base_url, json=payload, headers=headers, timeout=self.timeout
        )

        if response.status_code >= 500:
            msg = f"Fireworks Server: Error {response.status_code}"
            raise Exception(msg)
        if response.status_code >= 400:
            msg = f"Fireworks received an invalid payload: {response.text}"
            raise ValueError(msg)
        if response.status_code != 200:
            msg = (
                f"Fireworks returned an unexpected response with status "
                f"{response.status_code}: {response.text}"
            )
            raise Exception(msg)

        data = response.json()
        return self._format_output(data)

    async def _acall(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call Fireworks model to get predictions based on the prompt.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of strings to stop generation when encountered.
            run_manager: (Not used) Optional callback manager for async runs.
            kwargs: Additional parameters to pass to the model.

        Returns:
            The string generated by the model.

        """
        headers = {
            "Authorization": f"Bearer {self.fireworks_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        stop_to_use = stop[0] if stop and len(stop) == 1 else stop
        payload: dict[str, Any] = {
            **self.default_params,
            "prompt": prompt,
            "stop": stop_to_use,
            **kwargs,
        }

        # filter None values to not pass them to the http payload
        payload = {k: v for k, v in payload.items() if v is not None}
        async with (
            ClientSession() as session,
            session.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=ClientTimeout(total=self.timeout),
            ) as response,
        ):
            if response.status >= 500:
                msg = f"Fireworks Server: Error {response.status}"
                raise Exception(msg)
            if response.status >= 400:
                msg = f"Fireworks received an invalid payload: {response.text}"
                raise ValueError(msg)
            if response.status != 200:
                msg = (
                    f"Fireworks returned an unexpected response with status "
                    f"{response.status}: {response.text}"
                )
                raise Exception(msg)

            response_json = await response.json()
            return self._format_output(response_json)
