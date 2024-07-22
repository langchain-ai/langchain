import json
import logging
from typing import Any, Callable, Dict, List, Mapping, Optional

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Extra, SecretStr
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env, pre_init
from requests import ConnectTimeout, ReadTimeout, RequestException
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from langchain_community.llms.utils import enforce_stop_tokens

DEFAULT_NEBULA_SERVICE_URL = "https://api-nebula.symbl.ai"
DEFAULT_NEBULA_SERVICE_PATH = "/v1/model/generate"

logger = logging.getLogger(__name__)


class Nebula(LLM):
    """Nebula Service models.

    To use, you should have the environment variable ``NEBULA_SERVICE_URL``,
    ``NEBULA_SERVICE_PATH`` and ``NEBULA_API_KEY`` set with your Nebula
    Service, or pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_community.llms import Nebula

            nebula = Nebula(
                nebula_service_url="NEBULA_SERVICE_URL",
                nebula_service_path="NEBULA_SERVICE_PATH",
                nebula_api_key="NEBULA_API_KEY",
            )
    """

    """Key/value arguments to pass to the model. Reserved for future use"""
    model_kwargs: Optional[dict] = None

    """Optional"""

    nebula_service_url: Optional[str] = None
    nebula_service_path: Optional[str] = None
    nebula_api_key: Optional[SecretStr] = None
    model: Optional[str] = None
    max_new_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.95
    repetition_penalty: Optional[float] = 1.0
    top_k: Optional[int] = 1
    stop_sequences: Optional[List[str]] = None
    max_retries: Optional[int] = 10

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        nebula_service_url = get_from_dict_or_env(
            values,
            "nebula_service_url",
            "NEBULA_SERVICE_URL",
            DEFAULT_NEBULA_SERVICE_URL,
        )
        nebula_service_path = get_from_dict_or_env(
            values,
            "nebula_service_path",
            "NEBULA_SERVICE_PATH",
            DEFAULT_NEBULA_SERVICE_PATH,
        )
        nebula_api_key = convert_to_secret_str(
            get_from_dict_or_env(values, "nebula_api_key", "NEBULA_API_KEY", None)
        )

        if nebula_service_url.endswith("/"):
            nebula_service_url = nebula_service_url[:-1]
        if not nebula_service_path.startswith("/"):
            nebula_service_path = "/" + nebula_service_path

        values["nebula_service_url"] = nebula_service_url
        values["nebula_service_path"] = nebula_service_path
        values["nebula_api_key"] = nebula_api_key

        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Cohere API."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            "nebula_service_url": self.nebula_service_url,
            "nebula_service_path": self.nebula_service_path,
            **{"model_kwargs": _model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "nebula"

    def _invocation_params(
        self, stop_sequences: Optional[List[str]], **kwargs: Any
    ) -> dict:
        params = self._default_params
        if self.stop_sequences is not None and stop_sequences is not None:
            raise ValueError("`stop` found in both the input and default params.")
        elif self.stop_sequences is not None:
            params["stop_sequences"] = self.stop_sequences
        else:
            params["stop_sequences"] = stop_sequences
        return {**params, **kwargs}

    @staticmethod
    def _process_response(response: Any, stop: Optional[List[str]]) -> str:
        text = response["output"]["text"]
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
        """Call out to Nebula Service endpoint.
        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.
        Returns:
            The string generated by the model.
        Example:
            .. code-block:: python
                response = nebula("Tell me a joke.")
        """
        params = self._invocation_params(stop, **kwargs)
        prompt = prompt.strip()

        response = completion_with_retry(
            self,
            prompt=prompt,
            params=params,
            url=f"{self.nebula_service_url}{self.nebula_service_path}",
        )
        _stop = params.get("stop_sequences")
        return self._process_response(response, _stop)


def make_request(
    self: Nebula,
    prompt: str,
    url: str = f"{DEFAULT_NEBULA_SERVICE_URL}{DEFAULT_NEBULA_SERVICE_PATH}",
    params: Optional[Dict] = None,
) -> Any:
    """Generate text from the model."""
    params = params or {}
    api_key = None
    if self.nebula_api_key is not None:
        api_key = self.nebula_api_key.get_secret_value()
    headers = {
        "Content-Type": "application/json",
        "ApiKey": f"{api_key}",
    }

    body = {"prompt": prompt}

    # add params to body
    for key, value in params.items():
        body[key] = value

    # make request
    response = requests.post(url, headers=headers, json=body)

    if response.status_code != 200:
        raise Exception(
            f"Request failed with status code {response.status_code}"
            f" and message {response.text}"
        )

    return json.loads(response.text)


def _create_retry_decorator(llm: Nebula) -> Callable[[Any], Any]:
    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterward
    max_retries = llm.max_retries if llm.max_retries is not None else 3
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type((RequestException, ConnectTimeout, ReadTimeout))
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def completion_with_retry(llm: Nebula, **kwargs: Any) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(llm)

    @retry_decorator
    def _completion_with_retry(**_kwargs: Any) -> Any:
        return make_request(llm, **_kwargs)

    return _completion_with_retry(**kwargs)
