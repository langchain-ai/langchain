import os
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
    otherwise, an error is raised.  If the fine-tuned adapter is hosted at Predibase,
    then `adapter_version` in the adapter repository must be specified.

    An optional `predibase_sdk_version` parameter defaults to latest SDK version.
    """

    model: str
    predibase_api_key: SecretStr
    predibase_sdk_version: Optional[str] = None
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
        options: Dict[str, Union[str, float]] = {
            **(self.model_kwargs or {}),
            **self.default_options_for_generation,
            **(kwargs or {}),
        }
        if self._is_deprecated_sdk_version():
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

            base_llm_deployment: LLMDeployment = pc.LLM(
                uri=f"pb://deployments/{self.model}"
            )
            result: GeneratedResponse
            if self.adapter_id:
                """
                Attempt to retrieve the fine-tuned adapter from a Predibase
                repository.  If absent, then load the fine-tuned adapter
                from a HuggingFace repository.
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

        from predibase import Predibase

        os.environ["PREDIBASE_GATEWAY"] = "https://api.app.predibase.com"
        predibase: Predibase = Predibase(
            api_token=self.predibase_api_key.get_secret_value()
        )

        import requests
        from lorax.client import Client as LoraxClient
        from lorax.errors import GenerationError
        from lorax.types import Response

        lorax_client: LoraxClient = predibase.deployments.client(
            deployment_ref=self.model
        )

        response: Response
        if self.adapter_id:
            """
            Attempt to retrieve the fine-tuned adapter from a Predibase repository.
            If absent, then load the fine-tuned adapter from a HuggingFace repository.
            """
            if self.adapter_version:
                # Since the adapter version is provided, query the Predibase repository.
                pb_adapter_id: str = f"{self.adapter_id}/{self.adapter_version}"
                options.pop(
                    "api_token", None
                )  # The "api_token" is not used for Predibase-hosted models.
                try:
                    response = lorax_client.generate(
                        prompt=prompt,
                        adapter_id=pb_adapter_id,
                        **options,
                    )
                except GenerationError as ge:
                    raise ValueError(
                        f"""An adapter with the ID "{pb_adapter_id}" cannot be \
found in the Predibase repository of fine-tuned adapters."""
                    ) from ge
            else:
                # The adapter version is omitted,
                # hence look for the adapter ID in the HuggingFace repository.
                try:
                    response = lorax_client.generate(
                        prompt=prompt,
                        adapter_id=self.adapter_id,
                        adapter_source="hub",
                        **options,
                    )
                except GenerationError as ge:
                    raise ValueError(
                        f"""Either an adapter with the ID "{self.adapter_id}" \
cannot be found in a HuggingFace repository, or it is incompatible with the \
base model (please make sure that the adapter configuration is consistent).
"""
                    ) from ge
        else:
            try:
                response = lorax_client.generate(
                    prompt=prompt,
                    **options,
                )
            except requests.JSONDecodeError as jde:
                raise ValueError(
                    f"""An LLM with the deployment ID "{self.model}" cannot be found \
at Predibase (please refer to \
"https://docs.predibase.com/user-guide/inference/models" for the list of \
supported models).
"""
                ) from jde
        response_text = response.generated_text

        return response_text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"model_kwargs": self.model_kwargs},
        }

    def _is_deprecated_sdk_version(self) -> bool:
        try:
            import semantic_version
            from predibase.version import __version__ as current_version
            from semantic_version.base import Version

            sdk_semver_deprecated: Version = semantic_version.Version(
                version_string="2024.4.8"
            )
            actual_current_version: str = self.predibase_sdk_version or current_version
            sdk_semver_current: Version = semantic_version.Version(
                version_string=actual_current_version
            )
            return not (
                (sdk_semver_current > sdk_semver_deprecated)
                or ("+dev" in actual_current_version)
            )
        except ImportError as e:
            raise ImportError(
                "Could not import Predibase Python package. "
                "Please install it with `pip install semantic_version predibase`."
            ) from e
