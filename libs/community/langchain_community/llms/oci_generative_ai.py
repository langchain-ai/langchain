from __future__ import annotations

import json
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Iterator, List, Mapping, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator

from langchain_community.llms.utils import enforce_stop_tokens

CUSTOM_ENDPOINT_PREFIX = "ocid1.generativeaiendpoint"


class Provider(ABC):
    @property
    @abstractmethod
    def stop_sequence_key(self) -> str:
        ...

    @abstractmethod
    def completion_response_to_text(self, response: Any) -> str:
        ...


class CohereProvider(Provider):
    stop_sequence_key = "stop_sequences"

    def __init__(self) -> None:
        from oci.generative_ai_inference import models

        self.llm_inference_request = models.CohereLlmInferenceRequest

    def completion_response_to_text(self, response: Any) -> str:
        return response.data.inference_response.generated_texts[0].text


class MetaProvider(Provider):
    stop_sequence_key = "stop"

    def __init__(self) -> None:
        from oci.generative_ai_inference import models

        self.llm_inference_request = models.LlamaLlmInferenceRequest

    def completion_response_to_text(self, response: Any) -> str:
        return response.data.inference_response.choices[0].text


class OCIAuthType(Enum):
    """OCI authentication types as enumerator."""

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
    INSTANCE_PRINCIPAL, 
    RESOURCE_PRINCIPAL

    If not specified, API_KEY will be used
    """

    auth_profile: Optional[str] = "DEFAULT"
    """The name of the profile in ~/.oci/config
    If not specified , DEFAULT will be used 
    """

    model_id: str = None  # type: ignore[assignment]
    """Id of the model to call, e.g., cohere.command"""

    provider: str = None  # type: ignore[assignment]
    """Provider name of the model. Default to None, 
    will try to be derived from the model_id
    otherwise, requires user input
    """

    model_kwargs: Optional[Dict] = None
    """Keyword arguments to pass to the model"""

    service_endpoint: str = None  # type: ignore[assignment]
    """service endpoint url"""

    compartment_id: str = None  # type: ignore[assignment]
    """OCID of compartment"""

    is_stream: bool = False
    """Whether to stream back partial progress"""

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

                def make_security_token_signer(oci_config):  # type: ignore[no-untyped-def]
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
                raise ValueError(
                    "Please provide valid value to auth_type, "
                    f"{values['auth_type']} is not valid."
                )

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
                """Could not authenticate with OCI client. 
                Please check if ~/.oci/config exists.
                If INSTANCE_PRINCIPAL or RESOURCE_PRINCIPAL is used, 
                please check the specified
                auth_profile and auth_type are valid.""",
                e,
            ) from e

        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"model_kwargs": _model_kwargs},
        }

    def _get_provider(self, provider_map: Mapping[str, Any]) -> Any:
        if self.provider is not None:
            provider = self.provider
        else:
            provider = self.model_id.split(".")[0].lower()

        if provider not in provider_map:
            raise ValueError(
                f"Invalid provider derived from model_id: {self.model_id} "
                "Please explicitly pass in the supported provider "
                "when using custom endpoint"
            )
        return provider_map[provider]


class OCIGenAI(LLM, OCIGenAIBase):
    """OCI large language models.

    To authenticate, the OCI client uses the methods described in
    https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdk_authentication_methods.htm

    The authentifcation method is passed through auth_type and should be one of:
    API_KEY (default), SECURITY_TOKEN, INSTANCE_PRINCIPAL, RESOURCE_PRINCIPAL

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
        return "oci_generative_ai_completion"

    @property
    def _provider_map(self) -> Mapping[str, Any]:
        """Get the provider map"""
        return {
            "cohere": CohereProvider(),
            "meta": MetaProvider(),
        }

    @property
    def _provider(self) -> Any:
        """Get the internal provider object"""
        return self._get_provider(provider_map=self._provider_map)

    def _prepare_invocation_object(
        self, prompt: str, stop: Optional[List[str]], kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        from oci.generative_ai_inference import models

        _model_kwargs = self.model_kwargs or {}
        if stop is not None:
            _model_kwargs[self._provider.stop_sequence_key] = stop

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
            inference_request=self._provider.llm_inference_request(**inference_params),
        )

        return invocation_obj

    def _process_response(self, response: Any, stop: Optional[List[str]]) -> str:
        text = self._provider.completion_response_to_text(response)

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
        if self.is_stream:
            text = ""
            for chunk in self._stream(prompt, stop, run_manager, **kwargs):
                text += chunk.text
            if stop is not None:
                text = enforce_stop_tokens(text, stop)
            return text

        invocation_obj = self._prepare_invocation_object(prompt, stop, kwargs)
        response = self.client.generate_text(invocation_obj)
        return self._process_response(response, stop)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream OCIGenAI LLM on given prompt.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            An iterator of GenerationChunks.

        Example:
            .. code-block:: python

            response = llm.stream("Tell me a joke.")
        """

        self.is_stream = True
        invocation_obj = self._prepare_invocation_object(prompt, stop, kwargs)
        response = self.client.generate_text(invocation_obj)

        for event in response.data.events():
            json_load = json.loads(event.data)
            if "text" in json_load:
                event_data_text = json_load["text"]
            else:
                event_data_text = ""
            chunk = GenerationChunk(text=event_data_text)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk
