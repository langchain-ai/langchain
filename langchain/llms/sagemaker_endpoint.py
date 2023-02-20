"""Wrapper around Sagemaker InvokeEndpoint API."""
import json
from typing import Any, Callable, Dict, List, Mapping, Optional

from pydantic import BaseModel, Extra, root_validator

from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens


class SagemakerEndpoint(LLM, BaseModel):
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
                "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/abcdefghijklmnop/invocations"
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

    content_type: Optional[str] = "application/json"
    """The MIME type of the input data in the request body to be used in the header
    for the request to the Sagemaker invoke_endpoint API.
    Defaults to "application/json"."""

    model_input_transform_fn: Callable[[str, Dict], bytes]
    """
    Function which takes the prompt (str) and model_kwargs (dict) and transforms
    the input to the format which the model can accept as the request Body.
    Should return bytes or seekable file-like object in the format specified in the
    content_type request header.
    """

    """
    Example:
        .. code-block:: python

            def model_input_transform_fn(prompt, model_kwargs):
                parameter_payload = {"inputs": prompt, "parameters": model_kwargs}
                return json.dumps(parameter_payload).encode("utf-8")
    """

    model_kwargs: Optional[Dict] = None
    """Key word arguments to pass to the model."""

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
                "Please it install it with `pip install boto3`."
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
        if self.model_input_transform_fn is None:
            raise NotImplementedError("model_input_transform_fn not implemented")

        # send request
        try:
            response = self.client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                Body=self.model_input_transform_fn(prompt, _model_kwargs),
                ContentType=self.content_type,
            )
        except Exception as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

        response_json = json.loads(response["Body"].read().decode("utf-8"))
        text = response_json[0]["generated_text"]
        if stop is not None:
            # This is a bit hacky, but I can't figure out a better way to enforce
            # stop tokens when making calls to the sagemaker endpoint.
            text = enforce_stop_tokens(text, stop)

        return text
