import logging
from typing import Any, Dict, List, Mapping, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, LLMResult
from langchain_core.pydantic_v1 import Field, PrivateAttr, SecretStr
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env, pre_init

from langchain_community.llms.utils import enforce_stop_tokens

logger = logging.getLogger(__name__)


class CleanlabTLM(BaseLLM):
    """Cleanlab's Trustworthy Large Language Model.

    To use, you should have the ``cleanlab-studio`` python package installed,
    and the API key set either in the ``CLEANLAB_API_KEY`` environment variable,
    or pass it as a named parameter to the constructor.
    Sign up at app.cleanlab.ai to get a free API key.

    Example:
        .. code-block:: python

            from langchain_community.llms import CleanlabTLM
            tlm = CleanlabTLM(
                cleanlab_api_key="my_api_key",  # Not required if `CLEANLAB_API_KEY` env variable is set
                quality_preset="best"
            )
    """

    _client: Any = PrivateAttr()  # :meta private:

    cleanlab_api_key: Optional[SecretStr] = Field(default=None)
    """Cleanlab API key. Get it here: https://app.cleanlab.ai"""

    quality_preset: Optional[str] = Field(default="medium")
    """Presets to vary the quality of LLM response. Available presets listed here: 
        https://help.cleanlab.ai/reference/python/trustworthy_language_model/#class-tlmoptions
    """

    options: Optional[Dict[str, str]] = Field(default=None)
    """Holds configurations for trustworthy language model. 
       Available options (model, max_tokens, etc.) with their definitions listed here: 
       https://help.cleanlab.ai/reference/python/trustworthy_language_model/#class-tlmoptions
    """

    class Config:
        extra = "forbid"

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        cleanlab_api_key = convert_to_secret_str(
            get_from_dict_or_env(values, "cleanlab_api_key", "CLEANLAB_API_KEY")
        )
        values["cleanlab_api_key"] = cleanlab_api_key

        try:
            from cleanlab_studio import Studio

            studio = Studio(api_key=cleanlab_api_key.get_secret_value())
            # Check for user overrides in options dict
            use_options = values["options"] is not None
            # Initialize TLM
            cls._client = studio.TLM(
                quality_preset=values["quality_preset"],
                options=values["options"] if use_options else None,
            )
        except ImportError:
            raise ImportError(
                "Could not import cleanlab-studio python package. "
                "Please install it with `pip install -U cleanlab-studio`."
            )
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"quality_preset": self.quality_preset},
            **{"options": self.options},
        }

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Cleanlab API."""
        default_params = {
            "quality_preset": "medium",
            "max_tokens": 512,
            "model": "gpt-4o-mini",
        }
        return {**default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "cleanlab"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Call out to Cleanlab endpoint via client library and return response containing additional info."""

        responses: List[Dict[str, str]] = self._client.prompt(prompts)

        generations = []
        for resp in responses:
            text = resp["response"]
            trustworthiness_score = resp["trustworthiness_score"]
            if stop is not None:
                text = enforce_stop_tokens(text, stop)
            generations.append(
                [
                    Generation(
                        text=text,
                        generation_info={
                            "trustworthiness_score": trustworthiness_score
                        },
                    )
                ]
            )

        return LLMResult(generations=generations)

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Asynchronously call to Cleanlab endpoint."""

        responses: List[Dict[str, str]] = await self._client.prompt_async(prompts)

        generations = []
        for resp in responses:
            text = resp["response"]
            trustworthiness_score = resp["trustworthiness_score"]
            if stop is not None:
                text = enforce_stop_tokens(text, stop)
            generations.append(
                [
                    Generation(
                        text=text,
                        generation_info={
                            "trustworthiness_score": trustworthiness_score
                        },
                    )
                ]
            )

        return LLMResult(generations=generations)
