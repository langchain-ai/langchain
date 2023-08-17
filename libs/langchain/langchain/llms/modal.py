import logging
from typing import Any, Dict, List, Mapping, Optional

import requests

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.pydantic_v1 import Extra, Field, root_validator

logger = logging.getLogger(__name__)


class Modal(LLM):
    """Modal large language models.

    To use, you should have the ``modal-client`` python package installed.

    Any parameters that are valid to be passed to the call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain.llms import Modal
            modal = Modal(endpoint_url="")

    """

    endpoint_url: str = ""
    """model endpoint to use"""

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not
    explicitly specified."""

    class Config:
        """Configuration for this pydantic config."""

        extra = Extra.forbid

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = {field.alias for field in cls.__fields__.values()}

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
        return "modal"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call to Modal endpoint."""
        params = self.model_kwargs or {}
        params = {**params, **kwargs}
        response = requests.post(
            url=self.endpoint_url,
            headers={
                "Content-Type": "application/json",
            },
            json={"prompt": prompt, **params},
        )
        try:
            if prompt in response.json()["prompt"]:
                response_json = response.json()
        except KeyError:
            raise ValueError("LangChain requires 'prompt' key in response.")
        text = response_json["prompt"]
        if stop is not None:
            # I believe this is required since the stop tokens
            # are not enforced by the model parameters
            text = enforce_stop_tokens(text, stop)
        return text
