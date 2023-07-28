import json
import urllib.request
from abc import abstractmethod
from typing import Any, Dict, List, Mapping, Optional

from pydantic import BaseModel, validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.utils import get_from_dict_or_env


class AzureMLEndpointClient(object):
    """AzureML Managed Endpoint client."""

    def __init__(
        self, endpoint_url: str, endpoint_api_key: str, deployment_name: str
    ) -> None:
        """Initialize the class."""
        if not endpoint_api_key:
            raise ValueError("A key should be provided to invoke the endpoint")
        self.endpoint_url = endpoint_url
        self.endpoint_api_key = endpoint_api_key
        self.deployment_name = deployment_name

    def call(self, body: bytes) -> bytes:
        """call."""

        # The azureml-model-deployment header will force the request to go to a
        # specific deployment. Remove this header to have the request observe the
        # endpoint traffic rules.
        headers = {
            "Content-Type": "application/json",
            "Authorization": ("Bearer " + self.endpoint_api_key),
            "azureml-model-deployment": self.deployment_name,
        }

        req = urllib.request.Request(self.endpoint_url, body, headers)
        response = urllib.request.urlopen(req, timeout=50)
        result = response.read()
        return result


class ContentFormatterBase:
    """Transform request and response of AzureML endpoint to match with
    required schema.
    """

    """
    Example:
        .. code-block:: python
        
            class ContentFormatter(ContentFormatterBase):
                content_type = "application/json"
                accepts = "application/json"
                
                def format_request_payload(
                    self, 
                    prompt: str, 
                    model_kwargs: Dict
                ) -> bytes:
                    input_str = json.dumps(
                        {
                            "inputs": {"input_string": [prompt]}, 
                            "parameters": model_kwargs,
                        }
                    )
                    return str.encode(input_str)
                    
                def format_response_payload(self, output: str) -> str:
                    response_json = json.loads(output)
                    return response_json[0]["0"]
    """
    content_type: Optional[str] = "application/json"
    """The MIME type of the input data passed to the endpoint"""

    accepts: Optional[str] = "application/json"
    """The MIME type of the response data returned form the endpoint"""

    @abstractmethod
    def format_request_payload(self, prompt: str, model_kwargs: Dict) -> bytes:
        """Formats the request body according to the input schema of
        the model. Returns bytes or seekable file like object in the
        format specified in the content_type request header.
        """

    @abstractmethod
    def format_response_payload(self, output: bytes) -> str:
        """Formats the response body according to the output
        schema of the model. Returns the data type that is
        received from the response.
        """


class OSSContentFormatter(ContentFormatterBase):
    """Content handler for LLMs from the OSS catalog."""

    def format_request_payload(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps(
            {"inputs": {"input_string": [prompt]}, "parameters": model_kwargs}
        )
        return str.encode(input_str)

    def format_response_payload(self, output: bytes) -> str:
        response_json = json.loads(output)
        return response_json[0]["0"]


class HFContentFormatter(ContentFormatterBase):
    """Content handler for LLMs from the HuggingFace catalog."""

    def format_request_payload(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"inputs": [prompt], "parameters": model_kwargs})
        return str.encode(input_str)

    def format_response_payload(self, output: bytes) -> str:
        response_json = json.loads(output)
        return response_json[0][0]["generated_text"]


class DollyContentFormatter(ContentFormatterBase):
    """Content handler for the Dolly-v2-12b model"""

    def format_request_payload(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps(
            {"input_data": {"input_string": [prompt]}, "parameters": model_kwargs}
        )
        return str.encode(input_str)

    def format_response_payload(self, output: bytes) -> str:
        response_json = json.loads(output)
        return response_json[0]


class AzureMLOnlineEndpoint(LLM, BaseModel):
    """Azure ML Online Endpoint models.

    Example:
        .. code-block:: python

            azure_llm = AzureMLModel(
                endpoint_url="https://<your-endpoint>.<your_region>.inference.ml.azure.com/score",
                endpoint_api_key="my-api-key",
                deployment_name="my-deployment-name",
                content_formatter=content_formatter,
            )
    """  # noqa: E501

    endpoint_url: str = ""
    """URL of pre-existing Endpoint. Should be passed to constructor or specified as 
        env var `AZUREML_ENDPOINT_URL`."""

    endpoint_api_key: str = ""
    """Authentication Key for Endpoint. Should be passed to constructor or specified as
        env var `AZUREML_ENDPOINT_API_KEY`."""

    deployment_name: str = ""
    """Deployment Name for Endpoint. Should be passed to constructor or specified as
        env var `AZUREML_DEPLOYMENT_NAME`."""

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
        deployment_name = get_from_dict_or_env(
            values, "deployment_name", "AZUREML_DEPLOYMENT_NAME"
        )
        http_client = AzureMLEndpointClient(endpoint_url, endpoint_key, deployment_name)
        return http_client

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"deployment_name": self.deployment_name},
            **{"model_kwargs": _model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "azureml_endpoint"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> str:
        """Call out to an AzureML Managed Online endpoint.
        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.
        Returns:
            The string generated by the model.
        Example:
            .. code-block:: python
                response = azureml_model("Tell me a joke.")
        """
        _model_kwargs = self.model_kwargs or {}

        body = self.content_formatter.format_request_payload(prompt, _model_kwargs)
        endpoint_response = self.http_client.call(body)
        response = self.content_formatter.format_response_payload(endpoint_response)
        return response
