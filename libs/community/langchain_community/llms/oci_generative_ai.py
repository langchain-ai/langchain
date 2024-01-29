from __future__ import annotations

from abc import ABC
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator

from langchain_community.llms.utils import enforce_stop_tokens

CUSTOM_ENDPOINT_PREFIX = "ocid1.generativeaiendpoint"
VALID_PROVIDERS = ("cohere", "meta")


class OCIAuthType(Enum):
    API_KEY = 1
    SECURITY_TOKEN = 2
    INSTANCE_PRINCIPAL = 3
    RESOURCE_PRINCIPAL = 4


class OCIGenAIBase(BaseModel, ABC):
    """Base class for OCI GenAI models"""

    client: Any  #: :meta private:

    auth_type: Optional[str] = "API_KEY"
    """Authentication type, could be 
    
    API_KEY, 
    SECURITY_TOKEN, 
    INSTANCE_PRINCIPLE, 
    RESOURCE_PRINCIPLE

    If not specified, API_KEY will be used
    """

    auth_profile: Optional[str] = "DEFAULT"
    """The name of the profile in ~/.oci/config
    If not specified , DEFAULT will be used 
    """

    model_id: str = None
    """Id of the model to call, e.g., cohere.command"""

    provider: str = None
    """Provider name of the model. Default to None, 
    will try to be derived from the model_id
    otherwise, requires user input
    """

    model_kwargs: Optional[Dict] = None
    """Keyword arguments to pass to the model"""

    service_endpoint: str = None
    """service endpoint url"""

    compartment_id: str = None
    """OCID of compartment"""

    is_stream: bool = False
    """Whether to stream back partial progress"""

    llm_stop_sequence_mapping: Mapping[str, str] = {
        "cohere": "stop_sequences",
        "meta": "stop",
    }

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that OCI config and python package exists in environment."""

        # Skip creating new client if passed in constructor
        if values["client"] is not None:
            return values

        try:
            import oci

            client_kwargs = {
                "config": {},
                "signer": None,
                "service_endpoint": values["service_endpoint"],
                "retry_strategy": oci.retry.DEFAULT_RETRY_STRATEGY,
                "timeout": (10, 240),  # default timeout config for OCI Gen AI service
            }

            if values["auth_type"] == OCIAuthType(1).name:
                client_kwargs["config"] = oci.config.from_file(
                    profile_name=values["auth_profile"]
                )
                client_kwargs.pop("signer", None)
            elif values["auth_type"] == OCIAuthType(2).name:

                def make_security_token_signer(oci_config):
                    pk = oci.signer.load_private_key_from_file(
                        oci_config.get("key_file"), None
                    )
                    with open(
                        oci_config.get("security_token_file"), encoding="utf-8"
                    ) as f:
                        st_string = f.read()
                    return oci.auth.signers.SecurityTokenSigner(st_string, pk)

                client_kwargs["config"] = oci.config.from_file(
                    profile_name=values["auth_profile"]
                )
                client_kwargs["signer"] = make_security_token_signer(
                    oci_config=client_kwargs["config"]
                )
            elif values["auth_type"] == OCIAuthType(3).name:
                client_kwargs[
                    "signer"
                ] = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
            elif values["auth_type"] == OCIAuthType(4).name:
                client_kwargs[
                    "signer"
                ] = oci.auth.signers.get_resource_principals_signer()
            else:
                raise ValueError("Please provide valid value to auth_type")

            values["client"] = oci.generative_ai_inference.GenerativeAiInferenceClient(
                **client_kwargs
            )

        except ImportError as ex:
            raise ModuleNotFoundError(
                "Could not import oci python package. "
                "Please make sure you have the oci package installed."
            ) from ex
        except Exception as e:
            raise ValueError(
                "Could not authenticate with OCI client. "
                "Please check if ~/.oci/config exists. "
                "If INSTANCE_PRINCIPLE or RESOURCE_PRINCIPLE is used, "
                "Please check the specified "
                "auth_profile and auth_type are valid."
            ) from e

        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"model_kwargs": _model_kwargs},
        }

    def _get_provider(self) -> str:
        if self.provider is not None:
            provider = self.provider
        else:
            provider = self.model_id.split(".")[0].lower()

        if provider not in VALID_PROVIDERS:
            raise ValueError(
                f"Invalid provider derived from model_id: {self.model_id} "
                "Please explicitly pass in the supported provider "
                "when using custom endpoint"
            )
        return provider


class OCIGenAI(LLM, OCIGenAIBase):
    """OCI large language models.

    To authenticate, the OCI client uses the methods described in
    https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdk_authentication_methods.htm

    The authentifcation method is passed through auth_type and should be one of:
    API_KEY (default), SECURITY_TOKEN, INSTANCE_PRINCIPLE, RESOURCE_PRINCIPLE

    Make sure you have the required policies (profile/roles) to
    access the OCI Generative AI service.
    If a specific config profile is used, you must pass
    the name of the profile (from ~/.oci/config) through auth_profile.

    To use, you must provide the compartment id
    along with the endpoint url, and model id
    as named parameters to the constructor.

    Example:
        .. code-block:: python

            from langchain_community.llms import OCIGenAI

            llm = OCIGenAI(
                    model_id="MY_MODEL_ID",
                    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
                    compartment_id="MY_OCID"
                )
    """

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "oci"

    def _prepare_invocation_object(
        self, prompt: str, stop: Optional[List[str]], kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        from oci.generative_ai_inference import models

        oci_llm_request_mapping = {
            "cohere": models.CohereLlmInferenceRequest,
            "meta": models.LlamaLlmInferenceRequest,
        }
        provider = self._get_provider()
        _model_kwargs = self.model_kwargs or {}
        if stop is not None:
            _model_kwargs[self.llm_stop_sequence_mapping[provider]] = stop

        if self.model_id.startswith(CUSTOM_ENDPOINT_PREFIX):
            serving_mode = models.DedicatedServingMode(endpoint_id=self.model_id)
        else:
            serving_mode = models.OnDemandServingMode(model_id=self.model_id)

        inference_params = {**_model_kwargs, **kwargs}
        inference_params["prompt"] = prompt
        inference_params["is_stream"] = self.is_stream

        invocation_obj = models.GenerateTextDetails(
            compartment_id=self.compartment_id,
            serving_mode=serving_mode,
            inference_request=oci_llm_request_mapping[provider](**inference_params),
        )

        return invocation_obj

    def _process_response(self, response: Any, stop: Optional[List[str]]) -> str:
        provider = self._get_provider()
        if provider == "cohere":
            text = response.data.inference_response.generated_texts[0].text
        elif provider == "meta":
            text = response.data.inference_response.choices[0].text
        else:
            raise ValueError(f"Invalid provider: {provider}")

        if stop is not None:
            text = enforce_stop_tokens(text, stop)

        return text

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to OCIGenAI generate endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

               response = llm.invoke("Tell me a joke.")
        """

        invocation_obj = self._prepare_invocation_object(prompt, stop, kwargs)
        response = self.client.generate_text(invocation_obj)
        return self._process_response(response, stop)
