from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.utils import get_from_dict_or_env, pre_init
from langchain_core.utils.pydantic import get_fields
from pydantic import ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from replicate.prediction import Prediction

logger = logging.getLogger(__name__)


class Replicate(LLM):
    """Replicate models.

    To use, you should have the ``replicate`` python package installed,
    and the environment variable ``REPLICATE_API_TOKEN`` set with your API token.
    You can find your token here: https://replicate.com/account

    The model param is required, but any other model parameters can also
    be passed in with the format model_kwargs={model_param: value, ...}

    Example:
        .. code-block:: python

            from langchain_community.llms import Replicate

            replicate = Replicate(
                model=(
                    "stability-ai/stable-diffusion: "
                    "27b93a2413e7f36cd83da926f3656280b2931564ff050bf9575f1fdf9bcd7478",
                ),
                model_kwargs={"image_dimensions": "512x512"}
            )
    """

    model: str
    model_kwargs: Dict[str, Any] = Field(default_factory=dict, alias="input")
    replicate_api_token: Optional[str] = None
    prompt_key: Optional[str] = None
    version_obj: Any = Field(default=None, exclude=True)
    """Optionally pass in the model version object during initialization to avoid
        having to make an extra API call to retrieve it during streaming. NOTE: not
        serializable, is excluded from serialization.
    """

    streaming: bool = False
    """Whether to stream the results."""

    stop: List[str] = Field(default_factory=list)
    """Stop sequences to early-terminate generation."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"replicate_api_token": "REPLICATE_API_TOKEN"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "llms", "replicate"]

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: Dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = {field.alias for field in get_fields(cls).values()}

        input = values.pop("input", {})
        if input:
            logger.warning(
                "Init param `input` is deprecated, please use `model_kwargs` instead."
            )
        extra = {**values.pop("model_kwargs", {}), **input}
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
        replicate_api_token = get_from_dict_or_env(
            values, "replicate_api_token", "REPLICATE_API_TOKEN"
        )
        values["replicate_api_token"] = replicate_api_token
        return values

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "model_kwargs": self.model_kwargs,
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
        if self.streaming:
            completion: Optional[str] = None
            for chunk in self._stream(
                prompt, stop=stop, run_manager=run_manager, **kwargs
            ):
                if completion is None:
                    completion = chunk.text
                else:
                    completion += chunk.text
        else:
            prediction = self._create_prediction(prompt, **kwargs)
            prediction.wait()
            if prediction.status == "failed":
                raise RuntimeError(prediction.error)
            if isinstance(prediction.output, str):
                completion = prediction.output
            else:
                completion = "".join(prediction.output)
        assert completion is not None
        stop_conditions = stop or self.stop
        for s in stop_conditions:
            if s in completion:
                completion = completion[: completion.find(s)]
        return completion

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        prediction = self._create_prediction(prompt, **kwargs)
        stop_conditions = stop or self.stop
        stop_condition_reached = False
        current_completion: str = ""
        for output in prediction.output_iterator():
            current_completion += output
            # test for stop conditions, if specified
            for s in stop_conditions:
                if s in current_completion:
                    prediction.cancel()
                    stop_condition_reached = True
                    # Potentially some tokens that should still be yielded before ending
                    # stream.
                    stop_index = max(output.find(s), 0)
                    output = output[:stop_index]
                    if not output:
                        break
            if output:
                if run_manager:
                    run_manager.on_llm_new_token(
                        output,
                        verbose=self.verbose,
                    )
                yield GenerationChunk(text=output)
            if stop_condition_reached:
                break

    def _create_prediction(self, prompt: str, **kwargs: Any) -> Prediction:
        try:
            import replicate as replicate_python
        except ImportError:
            raise ImportError(
                "Could not import replicate python package. "
                "Please install it with `pip install replicate`."
            )

        # get the model and version
        if self.version_obj is None:
            if ":" in self.model:
                model_str, version_str = self.model.split(":")
                model = replicate_python.models.get(model_str)
                self.version_obj = model.versions.get(version_str)
            else:
                model = replicate_python.models.get(self.model)
                self.version_obj = model.latest_version

        if self.prompt_key is None:
            # sort through the openapi schema to get the name of the first input
            input_properties = sorted(
                self.version_obj.openapi_schema["components"]["schemas"]["Input"][
                    "properties"
                ].items(),
                key=lambda item: item[1].get("x-order", 0),
            )

            self.prompt_key = input_properties[0][0]

        input_: Dict = {
            self.prompt_key: prompt,
            **self.model_kwargs,
            **kwargs,
        }

        # if it's an official model
        if ":" not in self.model:
            return replicate_python.models.predictions.create(self.model, input=input_)
        else:
            return replicate_python.predictions.create(
                version=self.version_obj, input=input_
            )
