"""Wrapper around Sagemaker InvokeEndpoint API."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Union

from pydantic import Extra, root_validator

from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens


class ContentHandlerBase(ABC):
    """A handler class to transform input from LLM to a
    format that SageMaker endpoint expects. Similarily,
    the class also handles transforming output from the
    SageMaker endpoint to a format that LLM class expects.
    """

    """
    Example:
        .. code-block:: python

            class ContentHandler(ContentHandlerBase):
                content_type = "application/json"
                accepts = "application/json"

                def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
                    input_str = json.dumps({prompt: prompt, **model_kwargs})
                    return input_str.encode('utf-8')
                
                def transform_output(self, output: bytes) -> str:
                    response_json = json.loads(output.read().decode("utf-8"))
                    return response_json[0]["generated_text"]
    """

    content_type: Optional[str] = "text/plain"
    """The MIME type of the input data passed to endpoint"""

    accepts: Optional[str] = "text/plain"
    """The MIME type of the response data returned from endpoint"""

    @abstractmethod
    def transform_input(
        self, prompt: Union[str, List[str]], model_kwargs: Dict
    ) -> bytes:
        """Transforms the input to a format that model can accept
        as the request Body. Should return bytes or seekable file
        like object in the format specified in the content_type
        request header.
        """

    @abstractmethod
    def transform_output(self, output: bytes) -> Any:
        """Transforms the output from the model to string that
        the LLM class expects.
        """


class SagemakerEndpoint(LLM):
    """Wrapper around custom Sagemaker Inference Endpoints.

    To use, you must supply the endpoint name from your deployed
    Sagemaker model & the region where it is deployed.

    To authenticate, the AWS client uses the following methods to
    automatically load credentials:
    https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If a specific credential profile should be used, you must pass
    the name of the profile from the ~/.aws/credentials file that is to be used.

    Make sure the credentials / roles used have the required policies to
    access the Sagemaker endpoint.
    See: https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies.html
    """

    """
    Example:
        .. code-block:: python

            from langchain import SagemakerEndpoint
            endpoint_name = (
                "my-endpoint-name"
            )
            region_name = (
                "us-west-2"
            )
            credentials_profile_name = (
                "default"
            )
            se = SagemakerEndpoint(
                endpoint_name=endpoint_name,
                region_name=region_name,
                credentials_profile_name=credentials_profile_name
            )
    """
    client: Any  #: :meta private:

    endpoint_name: str = ""
    """The name of the endpoint from the deployed Sagemaker model.
    Must be unique within an AWS Region."""

    region_name: str = ""
    """The aws region where the Sagemaker model is deployed, eg. `us-west-2`."""

    credentials_profile_name: Optional[str] = None
    """The name of the profile in the ~/.aws/credentials or ~/.aws/config files, which
    has either access keys or role information specified.
    If not specified, the default credential profile or, if on an EC2 instance,
    credentials from IMDS will be used.
    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
    """

    content_handler: ContentHandlerBase
    """The content handler class that provides an input and
    output transform functions to handle formats between LLM
    and the endpoint.
    """

    """
     Example:
        .. code-block:: python

        class ContentHandler(ContentHandlerBase):
                content_type = "application/json"
                accepts = "application/json"

                def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
                    input_str = json.dumps({prompt: prompt, **model_kwargs})
                    return input_str.encode('utf-8')
                
                def transform_output(self, output: bytes) -> str:
                    response_json = json.loads(output.read().decode("utf-8"))
                    return response_json[0]["generated_text"]
    """

    model_kwargs: Optional[Dict] = None
    """Key word arguments to pass to the model."""

    endpoint_kwargs: Optional[Dict] = None
    """Optional attributes passed to the invoke_endpoint
    function. See `boto3`_. docs for more info.
    .. _boto3: <https://boto3.amazonaws.com/v1/documentation/api/latest/index.html>
    """

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that AWS credentials to and python package exists in environment."""
        try:
            import boto3

            try:
                if values["credentials_profile_name"] is not None:
                    session = boto3.Session(
                        profile_name=values["credentials_profile_name"]
                    )
                else:
                    # use default credentials
                    session = boto3.Session()

                values["client"] = session.client(
                    "sagemaker-runtime", region_name=values["region_name"]
                )

            except Exception as e:
                raise ValueError(
                    "Could not load credentials to authenticate with AWS client. "
                    "Please check that credentials in the specified "
                    "profile name are valid."
                ) from e

        except ImportError:
            raise ValueError(
                "Could not import boto3 python package. "
                "Please install it with `pip install boto3`."
            )
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"endpoint_name": self.endpoint_name},
            **{"model_kwargs": _model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "sagemaker_endpoint"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call out to Sagemaker inference endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = se("Tell me a joke.")
        """
        _model_kwargs = self.model_kwargs or {}
        _endpoint_kwargs = self.endpoint_kwargs or {}

        body = self.content_handler.transform_input(prompt, _model_kwargs)
        content_type = self.content_handler.content_type
        accepts = self.content_handler.accepts

        # send request
        try:
            response = self.client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                Body=body,
                ContentType=content_type,
                Accept=accepts,
                **_endpoint_kwargs,
            )
        except Exception as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

        text = self.content_handler.transform_output(response["Body"])
        if stop is not None:
            # This is a bit hacky, but I can't figure out a better way to enforce
            # stop tokens when making calls to the sagemaker endpoint.
            text = enforce_stop_tokens(text, stop)

        return text
