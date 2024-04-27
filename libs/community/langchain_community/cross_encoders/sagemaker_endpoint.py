import json
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator

from langchain_community.cross_encoders.base import BaseCrossEncoder


class CrossEncoderContentHandler:
    """Content handler for CrossEncoder class."""

    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, text_pairs: List[Tuple[str, str]]) -> bytes:
        input_str = json.dumps({"text_pairs": text_pairs})
        return input_str.encode("utf-8")

    def transform_output(self, output: Any) -> List[float]:
        response_json = json.loads(output.read().decode("utf-8"))
        scores = response_json["scores"]
        return scores


class SagemakerEndpointCrossEncoder(BaseModel, BaseCrossEncoder):
    """SageMaker Inference CrossEncoder endpoint.

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


           from langchain.embeddings import SagemakerEndpointCrossEncoder
           endpoint_name = (
               "my-endpoint-name"
           )
           region_name = (
               "us-west-2"
           )
           credentials_profile_name = (
               "default"
           )
           se = SagemakerEndpointCrossEncoder(
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

    content_handler: CrossEncoderContentHandler = CrossEncoderContentHandler()

    model_kwargs: Optional[Dict] = None
    """Keyword arguments to pass to the model."""

    endpoint_kwargs: Optional[Dict] = None
    """Optional attributes passed to the invoke_endpoint
   function. See `boto3`_. docs for more info.
   .. _boto3: <https://boto3.amazonaws.com/v1/documentation/api/latest/index.html>
   """

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

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
            raise ImportError(
                "Could not import boto3 python package. "
                "Please install it with `pip install boto3`."
            )
        return values

    def score(self, text_pairs: List[Tuple[str, str]]) -> List[float]:
        """Call out to SageMaker Inference CrossEncoder endpoint."""
        _endpoint_kwargs = self.endpoint_kwargs or {}

        body = self.content_handler.transform_input(text_pairs)
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

        return self.content_handler.transform_output(response["Body"])
