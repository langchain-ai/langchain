"""Wrapper around Replicate API."""
import logging
from typing import Any, Dict, List, Mapping, Optional

from pydantic import BaseModel, Extra, Field, root_validator

from langchain.llms.base import LLM
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class Replicate(LLM, BaseModel):
    """Wrapper around Replicate models.

    To use, you should have the ``replicate`` python package installed,
    and the environment variable ``REPLICATE_API_TOKEN`` set with your API token.
    You can find your token here: https://replicate.com/account

    model_name and model_version are required params, but any other model parameters
    can be passed in here.

    Example:
        .. code-block:: python
            from langchain.llms import Replicate
            replicate = Replicate(model=...)
    """

    model: str

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not
    explicitly specified."""

    replicate_api_token: Optional[str] = None

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
        replicate_api_token = get_from_dict_or_env(
            values, "REPLICATE_API_TOKEN", "REPLICATE_API_TOKEN"
        )
        values["replicate_api_token"] = replicate_api_token
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"model": self.model},
            **{"model_kwargs": self.model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        """Return type of model."""
        return "replicate"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call to replicate endpoint."""
        try:
            import replicate as replicate_python
        except ImportError:
            raise ValueError(
                "Could not import replicate python package. "
                "Please install it with `pip install replicate`."
            )
        params = self.model_kwargs or {}

        inputs = {"prompt": prompt, **params}

        outputs = replicate_python.run(self.model, input=inputs)
        return outputs[0]
