"""Wrapper around CerebriumAI API."""
import logging
from typing import Any, Dict, List, Mapping, Optional

from pydantic import BaseModel, Extra, Field, root_validator

from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class CerebriumAI(LLM, BaseModel):
    """Wrapper around CerebriumAI large language models.

    To use, you should have the ``cerebrium`` python package installed, and the
    environment variable ``CEREBRIUMAI_API_KEY`` set with your API key.

    Any parameters that are valid to be passed to the call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python
            from langchain.llms import CerebriumAI
            cerebrium = CerebriumAI(endpoint_url="")

    """

    endpoint_url: str = ""
    """model endpoint to use"""

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not
    explicitly specified."""

    cerebriumai_api_key: Optional[str] = None

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
                    f"""{field_name} was transfered to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)
        values["model_kwargs"] = extra
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        cerebriumai_api_key = get_from_dict_or_env(
            values, "cerebriumai_api_key", "CEREBRIUMAI_API_KEY"
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

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call to CerebriumAI endpoint."""
        try:
            from cerebrium import model_api_request
        except ImportError:
            raise ValueError(
                "Could not import cerebrium python package. "
                "Please install it with `pip install cerebrium`."
            )

        params = self.model_kwargs or {}
        response = model_api_request(
            self.endpoint_url, {"prompt": prompt, **params}, self.cerebriumai_api_key
        )
        text = response["data"]["result"]
        if stop is not None:
            # I believe this is required since the stop tokens
            # are not enforced by the model parameters
            text = enforce_stop_tokens(text, stop)
        return text
