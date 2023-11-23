from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from langchain_core.load.serializable import Serializable
from langchain_core.pydantic_v1 import Extra, Field, root_validator
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


def _create_retry_decorator(llm: Cohere) -> Callable[[Any], Any]:
    import cohere

    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(llm.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(retry_if_exception_type(cohere.error.CohereError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def completion_with_retry(llm: Cohere, **kwargs: Any) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(llm)

    @retry_decorator
    def _completion_with_retry(**kwargs: Any) -> Any:
        return llm.client.generate(**kwargs)

    return _completion_with_retry(**kwargs)


def acompletion_with_retry(llm: Cohere, **kwargs: Any) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(llm)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        return await llm.async_client.generate(**kwargs)

    return _completion_with_retry(**kwargs)


class BaseCohere(Serializable):
    """Base class for Cohere models."""

    client: Any  #: :meta private:
    async_client: Any  #: :meta private:
    model: Optional[str] = Field(default=None)
    """Model name to use."""

    temperature: float = 0.75
    """A non-negative float that tunes the degree of randomness in generation."""

    cohere_api_key: Optional[str] = None

    stop: Optional[List[str]] = None

    streaming: bool = Field(default=False)
    """Whether to stream the results."""

    user_agent: str = "langchain"
    """Identifier for the application making the request."""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "Could not import cohere python package. "
                "Please install it with `pip install cohere`."
            )
        else:
            cohere_api_key = get_from_dict_or_env(
                values, "cohere_api_key", "COHERE_API_KEY"
            )
            client_name = values["user_agent"]
            values["client"] = cohere.Client(cohere_api_key, client_name=client_name)
            values["async_client"] = cohere.AsyncClient(
                cohere_api_key, client_name=client_name
            )
        return values


class Cohere(LLM, BaseCohere):
    """Cohere large language models.

    To use, you should have the ``cohere`` python package installed, and the
    environment variable ``COHERE_API_KEY`` set with your API key, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain.llms import Cohere

            cohere = Cohere(model="gptd-instruct-tft", cohere_api_key="my-api-key")
    """

    max_tokens: int = 256
    """Denotes the number of tokens to predict per generation."""

    k: int = 0
    """Number of most likely tokens to consider at each step."""

    p: int = 1
    """Total probability mass of tokens to consider at each step."""

    frequency_penalty: float = 0.0
    """Penalizes repeated tokens according to frequency. Between 0 and 1."""

    presence_penalty: float = 0.0
    """Penalizes repeated tokens. Between 0 and 1."""

    truncate: Optional[str] = None
    """Specify how the client handles inputs longer than the maximum token
    length: Truncate from START, END or NONE"""

    max_retries: int = 10
    """Maximum number of retries to make when generating."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Cohere API."""
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "k": self.k,
            "p": self.p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "truncate": self.truncate,
        }

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"cohere_api_key": "COHERE_API_KEY"}

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{"model": self.model}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "cohere"

    def _invocation_params(self, stop: Optional[List[str]], **kwargs: Any) -> dict:
        params = self._default_params
        if self.stop is not None and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")
        elif self.stop is not None:
            params["stop_sequences"] = self.stop
        else:
            params["stop_sequences"] = stop
        return {**params, **kwargs}

    def _process_response(self, response: Any, stop: Optional[List[str]]) -> str:
        text = response.generations[0].text
        # If stop tokens are provided, Cohere's endpoint returns them.
        # In order to make this consistent with other endpoints, we strip them.
        if stop:
            text = enforce_stop_tokens(text, stop)
        return text

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to Cohere's generate endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = cohere("Tell me a joke.")
        """
        params = self._invocation_params(stop, **kwargs)
        response = completion_with_retry(
            self, model=self.model, prompt=prompt, **params
        )
        _stop = params.get("stop_sequences")
        return self._process_response(response, _stop)

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Async call out to Cohere's generate endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = await cohere("Tell me a joke.")
        """
        params = self._invocation_params(stop, **kwargs)
        response = await acompletion_with_retry(
            self, model=self.model, prompt=prompt, **params
        )
        _stop = params.get("stop_sequences")
        return self._process_response(response, _stop)
