from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Literal, Optional

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Field, SecretStr
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env

from langchain_community.llms.utils import enforce_stop_tokens

logger = logging.getLogger(__name__)


class YiLLM(LLM):
    """Yi large language models."""

    model: str = "yi-large"
    temperature: float = 0.3
    top_p: float = 0.95
    timeout: int = 60
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    yi_api_key: Optional[SecretStr] = None
    region: Literal["auto", "domestic", "international"] = "auto"
    yi_api_url_domestic: str = "https://api.lingyiwanwu.com/v1/chat/completions"
    yi_api_url_international: str = "https://api.01.ai/v1/chat/completions"

    def __init__(self, **kwargs: Any):
        kwargs["yi_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(kwargs, "yi_api_key", "YI_API_KEY")
        )
        super().__init__(**kwargs)

    @property
    def _default_params(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            **self.model_kwargs,
        }

    def _post(self, request: Any) -> Any:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.yi_api_key.get_secret_value()}",  # type: ignore
        }

        urls = []
        if self.region == "domestic":
            urls = [self.yi_api_url_domestic]
        elif self.region == "international":
            urls = [self.yi_api_url_international]
        else:  # auto
            urls = [self.yi_api_url_domestic, self.yi_api_url_international]

        for url in urls:
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=request,
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    parsed_json = json.loads(response.text)
                    return parsed_json["choices"][0]["message"]["content"]
                elif (
                    response.status_code != 403
                ):  # If not a permission error, raise immediately
                    response.raise_for_status()
            except requests.RequestException as e:
                if url == urls[-1]:  # If this is the last URL to try
                    raise ValueError(f"An error has occurred: {e}")
                else:
                    logger.warning(f"Failed to connect to {url}, trying next URL")
                    continue

        raise ValueError("Failed to connect to all available URLs")

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        request = self._default_params
        request["messages"] = [{"role": "user", "content": prompt}]
        request.update(kwargs)
        text = self._post(request)
        if stop is not None:
            text = enforce_stop_tokens(text, stop)
        return text

    @property
    def _llm_type(self) -> str:
        """Return type of chat_model."""
        return "yi-llm"
