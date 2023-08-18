import logging
from typing import Any, Dict, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.pydantic_v1 import Extra, Field, root_validator
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class Replicate(LLM):
    """Replicate models.

    To use, you should have the ``replicate`` python package installed,
    and the environment variable ``REPLICATE_API_TOKEN`` set with your API token.
    You can find your token here: https://replicate.com/account

    The model param is required, but any other model parameters can also
    be passed in with the format input={model_param: value, ...}

    Example:
        .. code-block:: python

            from langchain.llms import Replicate
            replicate = Replicate(model="stability-ai/stable-diffusion: \
                                         27b93a2413e7f36cd83da926f365628\
                                         0b2931564ff050bf9575f1fdf9bcd7478",
                                  input={"image_dimensions": "512x512"})
    """

    model: str
    input: Dict[str, Any] = Field(default_factory=dict)
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    replicate_api_token: Optional[str] = None

    streaming: bool = Field(default=False)
    """Whether to stream the results."""

    stop: Optional[List[str]] = Field(default=[])
    """Stop sequences to early-terminate generation."""

    class Config:
        """Configuration for this pydantic config."""

        extra = Extra.forbid

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"replicate_api_token": "REPLICATE_API_TOKEN"}

    @property
    def lc_serializable(self) -> bool:
        return True

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
            "model": self.model,
            **{"model_kwargs": self.model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        """Return type of model."""
        return "replicate"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call to replicate endpoint."""
        try:
            import replicate as replicate_python
        except ImportError:
            raise ImportError(
                "Could not import replicate python package. "
                "Please install it with `pip install replicate`."
            )

        # get the model and version
        model_str, version_str = self.model.split(":")
        model = replicate_python.models.get(model_str)
        version = model.versions.get(version_str)

        # sort through the openapi schema to get the name of the first input
        input_properties = sorted(
            version.openapi_schema["components"]["schemas"]["Input"][
                "properties"
            ].items(),
            key=lambda item: item[1].get("x-order", 0),
        )
        first_input_name = input_properties[0][0]
        inputs = {first_input_name: prompt, **self.input}

        prediction = replicate_python.predictions.create(
            version=version, input={**inputs, **kwargs}
        )
        current_completion: str = ""
        stop_condition_reached = False
        for output in prediction.output_iterator():
            current_completion += output

            # test for stop conditions, if specified
            if stop:
                for s in stop:
                    if s in current_completion:
                        prediction.cancel()
                        stop_index = current_completion.find(s)
                        current_completion = current_completion[:stop_index]
                        stop_condition_reached = True
                        break

            if stop_condition_reached:
                break

            if self.streaming and run_manager:
                run_manager.on_llm_new_token(output)
        return current_completion
