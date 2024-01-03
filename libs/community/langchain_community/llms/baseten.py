import logging
import os
from typing import Any, Dict, List, Mapping, Optional

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Field

logger = logging.getLogger(__name__)


class Baseten(LLM):
    """Baseten model

    This module allows using LLMs hosted on Baseten.

    The LLM deployed on Baseten must have the following properties:

    * Must accept input as a dictionary with the key "prompt"
    * May accept other input in the dictionary passed through with kwargs
    * Must return a string with the model output

    To use this module, you must:

    * Export your Baseten API key as the environment variable `BASETEN_API_KEY`
    * Get the model ID for your model from your Baseten dashboard
    * Identify the model deployment ("production" for all model library models)

    These code samples use
    [Mistral 7B Instruct](https://app.baseten.co/explore/mistral_7b_instruct)
    from Baseten's model library.

    Examples:
        .. code-block:: python

            from langchain_community.llms import Baseten
            # Production deployment
            mistral = Baseten(model="MODEL_ID", deployment="production")
            mistral("What is the Mistral wind?")

        .. code-block:: python

            from langchain_community.llms import Baseten
            # Development deployment
            mistral = Baseten(model="MODEL_ID", deployment="development")
            mistral("What is the Mistral wind?")

        .. code-block:: python

            from langchain_community.llms import Baseten
            # Other published deployment
            mistral = Baseten(model="MODEL_ID", deployment="DEPLOYMENT_ID")
            mistral("What is the Mistral wind?")
    """

    model: str
    deployment: str
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
        baseten_api_key = os.environ["BASETEN_API_KEY"]
        model_id = self.model
        if self.deployment == "production":
            model_url = f"https://model-{model_id}.api.baseten.co/production/predict"
        elif self.deployment == "development":
            model_url = f"https://model-{model_id}.api.baseten.co/development/predict"
        else:  # try specific deployment ID
            model_url = f"https://model-{model_id}.api.baseten.co/deployment/{self.deployment}/predict"
        response = requests.post(
            model_url,
            headers={"Authorization": f"Api-Key {baseten_api_key}"},
            json={"prompt": prompt, **kwargs},
        )
        return response.json()
