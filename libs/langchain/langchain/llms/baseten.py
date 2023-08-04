import logging
from typing import Any, Dict, List, Mapping, Optional

from pydantic import Field

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

logger = logging.getLogger(__name__)


class Baseten(LLM):
    """Baseten models.

    To use, you should have the ``baseten`` python package installed,
    and run ``baseten.login()`` with your Baseten API key.

    The required ``model`` param can be either a model id or model
    version id. Using a model version ID will result in
    slightly faster invocation.
    Any other model parameters can also
    be passed in with the format input={model_param: value, ...}

    The Baseten model must accept a dictionary of input with the key
    "prompt" and return a dictionary with a key "data" which maps
    to a list of response strings.

    Example:
        .. code-block:: python
            from langchain.llms import Baseten
            my_model = Baseten(model="MODEL_ID")
            output = my_model("prompt")
    """

    model: str
    input: Dict[str, Any] = Field(default_factory=dict)
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"model_kwargs": self.model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        """Return type of model."""
        return "baseten"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call to Baseten deployed model endpoint."""
        try:
            import baseten
        except ImportError as exc:
            raise ImportError(
                "Could not import Baseten Python package. "
                "Please install it with `pip install baseten`."
            ) from exc

        # get the model and version
        try:
            model = baseten.deployed_model_version_id(self.model)
            response = model.predict({"prompt": prompt, **kwargs})
        except baseten.common.core.ApiError:
            model = baseten.deployed_model_id(self.model)
            response = model.predict({"prompt": prompt, **kwargs})
        return "".join(response)
