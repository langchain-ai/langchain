import logging
from typing import Any, Dict, List, Mapping, Optional, cast

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Extra, Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env, pre_init

from langchain_community.llms.utils import enforce_stop_tokens

logger = logging.getLogger(__name__)


class Banana(LLM):
    """Banana large language models.

    To use, you should have the ``banana-dev`` python package installed,
    and the environment variable ``BANANA_API_KEY`` set with your API key.
    This is the team API key available in the Banana dashboard.

    Any parameters that are valid to be passed to the call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain_community.llms import Banana
            banana = Banana(model_key="", model_url_slug="")
    """

    model_key: str = ""
    """model key to use"""

    model_url_slug: str = ""
    """model endpoint to use"""

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not
    explicitly specified."""

    banana_api_key: Optional[SecretStr] = None

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

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        banana_api_key = convert_to_secret_str(
            get_from_dict_or_env(values, "banana_api_key", "BANANA_API_KEY")
        )
        values["banana_api_key"] = banana_api_key
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"model_key": self.model_key},
            **{"model_url_slug": self.model_url_slug},
            **{"model_kwargs": self.model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "bananadev"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call to Banana endpoint."""
        try:
            from banana_dev import Client
        except ImportError:
            raise ImportError(
                "Could not import banana-dev python package. "
                "Please install it with `pip install banana-dev`."
            )
        params = self.model_kwargs or {}
        params = {**params, **kwargs}
        api_key = cast(SecretStr, self.banana_api_key)
        model_key = self.model_key
        model_url_slug = self.model_url_slug
        model_inputs = {
            # a json specific to your model.
            "prompt": prompt,
            **params,
        }
        model = Client(
            # Found in main dashboard
            api_key=api_key.get_secret_value(),
            # Both found in model details page
            model_key=model_key,
            url=f"https://{model_url_slug}.run.banana.dev",
        )
        response, meta = model.call("/", model_inputs)
        try:
            text = response["outputs"]
        except (KeyError, TypeError):
            raise ValueError(
                "Response should be of schema: {'outputs': 'text'}."
                "\nTo fix this:"
                "\n- fork the source repo of the Banana model"
                "\n- modify app.py to return the above schema"
                "\n- deploy that as a custom repo"
            )
        if stop is not None:
            # I believe this is required since the stop tokens
            # are not enforced by the model parameters
            text = enforce_stop_tokens(text, stop)
        return text
