import logging
from typing import Any, Dict, List, Mapping, Optional, cast

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env, pre_init
from pydantic import ConfigDict, Field, SecretStr, model_validator

from langchain_community.llms.utils import enforce_stop_tokens

logger = logging.getLogger(__name__)


class CerebriumAI(LLM):
    """CerebriumAI large language models.

    To use, you should have the ``cerebrium`` python package installed.
    You should also have the environment variable ``CEREBRIUMAI_API_KEY``
    set with your API key or pass it as a named argument in the constructor.

    Any parameters that are valid to be passed to the call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain_community.llms import CerebriumAI
            cerebrium = CerebriumAI(endpoint_url="", cerebriumai_api_key="my-api-key")

    """

    endpoint_url: str = ""
    """model endpoint to use"""

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not
    explicitly specified."""

    cerebriumai_api_key: Optional[SecretStr] = None

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: Dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = set(list(cls.model_fields.keys()))

        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name not in all_required_field_names:
                if field_name in extra:
                    raise ValueError(f"Found {field_name} supplied twice.")
                logger.warning(
                    f"""{field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)
        values["model_kwargs"] = extra
        return values

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        cerebriumai_api_key = convert_to_secret_str(
            get_from_dict_or_env(values, "cerebriumai_api_key", "CEREBRIUMAI_API_KEY")
        )
        values["cerebriumai_api_key"] = cerebriumai_api_key
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"endpoint_url": self.endpoint_url},
            **{"model_kwargs": self.model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "cerebriumai"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        headers: Dict = {
            "Authorization": cast(
                SecretStr, self.cerebriumai_api_key
            ).get_secret_value(),
            "Content-Type": "application/json",
        }
        params = self.model_kwargs or {}
        payload = {"prompt": prompt, **params, **kwargs}
        response = requests.post(self.endpoint_url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            text = data["result"]
            if stop is not None:
                # I believe this is required since the stop tokens
                # are not enforced by the model parameters
                text = enforce_stop_tokens(text, stop)
            return text
        else:
            response.raise_for_status()
        return ""
