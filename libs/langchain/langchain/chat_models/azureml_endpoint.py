import urllib.request

from typing import Any, Optional, Mapping, List, Dict

from pydantic import BaseModel, validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.schema import ChatResult
from langchain.schema.messages import BaseMessage
from langchain.utils import get_from_dict_or_env

class AzureMLEndpointClient(object):
    """AzureML Managed Endpoint client."""

    def __init__(
        self, endpoint_url: str, endpoint_api_key: str, deployment_name: str = ""
    ) -> None:
        """Initialize the class."""
        if not endpoint_api_key or not endpoint_url:
            raise ValueError("A key/token and REST endpoint should be provided to invoke the endpoint")
        self.endpoint_url = endpoint_url
        self.endpoint_api_key = endpoint_api_key
        self.deployment_name = deployment_name

    def call(self, body: bytes, **kwargs: Any) -> bytes:
        """call."""

        # The azureml-model-deployment header will force the request to go to a
        # specific deployment. Remove this header to have the request observe the
        # endpoint traffic rules.
        headers = {
            "Content-Type": "application/json",
            "Authorization": ("Bearer " + self.endpoint_api_key),
        }
        if self.deployment_name != "":
            headers["azureml-model-deployment"] = self.deployment_name

        req = urllib.request.Request(self.endpoint_url, body, headers)
        response = urllib.request.urlopen(req, timeout=kwargs.get("timeout", 50))
        result = response.read()
        return result

class AzureMLChatOnlineEndpoint(LLM, BaseModel):
    endpoint_url: str = ""
    """URL of pre-existing Endpoint. Should be passed to constructor or specified as 
        env var `AZUREML_ENDPOINT_URL`."""

    endpoint_api_key: str = ""
    """Authentication Key for Endpoint. Should be passed to constructor or specified as
        env var `AZUREML_ENDPOINT_API_KEY`."""
    
    http_client: Any = None  #: :meta private:

    content_formatter: Any = None
    """The content formatter that provides an input and output
    transform function to handle formats between the LLM and
    the endpoint"""

    model_kwargs: Optional[dict] = None
    """Key word arguments to pass to the model."""

    
    @validator("http_client", always=True, allow_reuse=True)
    @classmethod
    def validate_client(cls, field_value: Any, values: Dict) -> AzureMLEndpointClient:
        """Validate that api key and python package exists in environment."""
        endpoint_key = get_from_dict_or_env(
            values, "endpoint_api_key", "AZUREML_ENDPOINT_API_KEY"
        )
        endpoint_url = get_from_dict_or_env(
            values, "endpoint_url", "AZUREML_ENDPOINT_URL"
        )
        http_client = AzureMLEndpointClient(endpoint_url, endpoint_key)
        return http_client
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"model_kwargs": _model_kwargs},
        }
    
    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "azureml_chat_endpoint"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        pass