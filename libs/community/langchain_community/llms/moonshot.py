from typing import Any, Dict, List, Optional

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env

from langchain_community.llms.utils import enforce_stop_tokens

MOONSHOT_SERVICE_URL_BASE = "https://api.moonshot.cn/v1"


class _MoonshotClient(BaseModel):
    """An API client that talks to the Moonshot server."""

    api_key: SecretStr
    """The API key to use for authentication."""
    base_url: str = MOONSHOT_SERVICE_URL_BASE

    def completion(self, request: Any) -> Any:
        headers = {"Authorization": f"Bearer {self.api_key.get_secret_value()}"}
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=request,
        )
        if not response.ok:
            raise ValueError(f"HTTP {response.status_code} error: {response.text}")
        return response.json()["choices"][0]["message"]["content"]


class MoonshotCommon(BaseModel):
    _client: _MoonshotClient
    base_url: str = MOONSHOT_SERVICE_URL_BASE
    moonshot_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    """Moonshot API key. Get it here: https://platform.moonshot.cn/console/api-keys"""
    model_name: str = Field(default="moonshot-v1-8k", alias="model")
    """Model name. Available models listed here: https://platform.moonshot.cn/pricing"""
    max_tokens = 1024
    """Maximum number of tokens to generate."""
    temperature = 0.3
    """Temperature parameter (higher values make the model more creative)."""

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    @property
    def lc_secrets(self) -> dict:
        """A map of constructor argument names to secret ids.

        For example,
            {"moonshot_api_key": "MOONSHOT_API_KEY"}
        """
        return {"moonshot_api_key": "MOONSHOT_API_KEY"}

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        return {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        return {**{"model": self.model_name}, **self._default_params}

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra parameters.
        Override the superclass method, prevent the model parameter from being
        overridden.
        """
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["moonshot_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "moonshot_api_key", "MOONSHOT_API_KEY")
        )

        values["_client"] = _MoonshotClient(
            api_key=values["moonshot_api_key"],
            base_url=values["base_url"]
            if "base_url" in values
            else MOONSHOT_SERVICE_URL_BASE,
        )
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "moonshot"


class Moonshot(MoonshotCommon, LLM):
    """Moonshot large language models.

    To use, you should have the environment variable ``MOONSHOT_API_KEY`` set with your
    API key. Referenced from https://platform.moonshot.cn/docs

    Example:
        .. code-block:: python

            from langchain_community.llms.moonshot import Moonshot

            moonshot = Moonshot(model="moonshot-v1-8k")
    """

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        request = self._invocation_params
        request["messages"] = [{"role": "user", "content": prompt}]
        request.update(kwargs)
        text = self._client.completion(request)
        if stop is not None:
            # This is required since the stop tokens
            # are not enforced by the model parameters
            text = enforce_stop_tokens(text, stop)

        return text
