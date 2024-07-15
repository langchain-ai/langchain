from typing import Any, Dict, List, Optional

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env, pre_init

from langchain_community.llms.utils import enforce_stop_tokens

YI_SERVICE_URL_DOMESTIC = "https://api.lingyiwanwu.com/v1"
YI_SERVICE_URL_INTERNATIONAL = "https://api.01.ai/v1"


class _YiClient(BaseModel):
    """An API client that talks to the Yi server."""

    api_key: SecretStr
    """The API key to use for authentication."""
    base_url: str = YI_SERVICE_URL_INTERNATIONAL

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


class YiCommon(BaseModel):
    """Common parameters for Yi LLMs."""

    _client: _YiClient
    base_url: str = YI_SERVICE_URL_INTERNATIONAL
    yi_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    """Yi API key. Get it from the Yi platform.https://platform.01.ai/apikeys or https://platform.lingyiwanwu.com/apikeys"""
    model_name: str = Field(default="yi-large", alias="model")
    """Model name. Default is yi-large."""
    max_tokens: int = 1024
    """Maximum number of tokens to generate."""
    temperature: float = 0.7
    """Temperature parameter (higher values make the model more creative)."""

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    @property
    def lc_secrets(self) -> dict:
        """A map of constructor argument names to secret ids."""
        return {"yi_api_key": "YI_API_KEY"}

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Yi API."""
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
        """Build extra parameters."""
        return values

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["yi_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "yi_api_key", "YI_API_KEY")
        )

        # Determine the base URL based on the provided value or default to international
        base_url = values.get("base_url", YI_SERVICE_URL_INTERNATIONAL)
        if base_url not in [YI_SERVICE_URL_DOMESTIC, YI_SERVICE_URL_INTERNATIONAL]:
            raise ValueError("Invalid base_url. Must be either domestic or international Yi API URL.")

        values["_client"] = _YiClient(
            api_key=values["yi_api_key"],
            base_url=base_url,
        )
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "yi"


class Yi(YiCommon, LLM):
    """Yi large language models.

    To use, you should have the environment variable ``YI_API_KEY`` set with your
    API key. Referenced from https://platform.01.ai/docs

    Example:
        .. code-block:: python

            from langchain_community.llms.yi import Yi

            yi = Yi(model="yi-large")
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
            text = enforce_stop_tokens(text, stop)

        return text
