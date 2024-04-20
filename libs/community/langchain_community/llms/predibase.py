from typing import Any, Dict, List, Mapping, Optional, Union

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Field, SecretStr


class Predibase(LLM):
    """Use your Predibase models with Langchain.

    To use, you should have the ``predibase`` python package installed,
    and have your Predibase API key.

    The `model` parameter is the Predibase "serverless" base_model ID
    (see https://docs.predibase.com/user-guide/inference/models for the catalog).

    An optional `adapter_id` parameter is the Predibase ID or HuggingFace ID of a
    fine-tuned LLM adapter, whose base model is the `model` parameter; the
    fine-tuned adapter must be compatible with its base model;
    otherwise, an error is raised.  If a Predibase ID references the
    fine-tuned adapter, then the `adapter_version` in the adapter repository can
    be optionally specified; omitting it defaults to the most recent version.
    """

    model: str
    predibase_api_key: SecretStr
    adapter_id: Optional[str] = None
    adapter_version: Optional[int] = None
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    default_options_for_generation: dict = Field(
        {
            "max_new_tokens": 256,
            "temperature": 0.1,
        },
        const=True,
    )

    @property
    def _llm_type(self) -> str:
        return "predibase"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            from predibase import PredibaseClient
            from predibase.pql import get_session
            from predibase.pql.api import (
                ServerResponseError,
                Session,
            )
            from predibase.resource.llm.interface import (
                HuggingFaceLLM,
                LLMDeployment,
            )
            from predibase.resource.llm.response import GeneratedResponse
            from predibase.resource.model import Model

            session: Session = get_session(
                token=self.predibase_api_key.get_secret_value(),
                gateway="https://api.app.predibase.com/v1",
                serving_endpoint="serving.app.predibase.com",
            )
            pc: PredibaseClient = PredibaseClient(session=session)
        except ImportError as e:
            raise ImportError(
                "Could not import Predibase Python package. "
                "Please install it with `pip install predibase`."
            ) from e
        except ValueError as e:
            raise ValueError("Your API key is not correct. Please try again") from e
        options: Dict[str, Union[str, float]] = (
            self.model_kwargs or self.default_options_for_generation
        )
        base_llm_deployment: LLMDeployment = pc.LLM(
            uri=f"pb://deployments/{self.model}"
        )
        result: GeneratedResponse
        if self.adapter_id:
            """
            Attempt to retrieve the fine-tuned adapter from a Predibase repository.
            If absent, then load the fine-tuned adapter from a HuggingFace repository.
            """
            adapter_model: Union[Model, HuggingFaceLLM]
            try:
                adapter_model = pc.get_model(
                    name=self.adapter_id,
                    version=self.adapter_version,
                    model_id=None,
                )
            except ServerResponseError:
                # Predibase does not recognize the adapter ID (query HuggingFace).
                adapter_model = pc.LLM(uri=f"hf://{self.adapter_id}")
            result = base_llm_deployment.with_adapter(model=adapter_model).generate(
                prompt=prompt,
                options=options,
            )
        else:
            result = base_llm_deployment.generate(
                prompt=prompt,
                options=options,
            )
        return result.response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"model_kwargs": self.model_kwargs},
        }
