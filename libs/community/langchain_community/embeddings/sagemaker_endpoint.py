from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra
from langchain_core.utils import pre_init

from langchain_community.llms.sagemaker_endpoint import ContentHandlerBase


class EmbeddingsContentHandler(ContentHandlerBase[List[str], List[List[float]]]):
    """Content handler for LLM class."""


class SagemakerEndpointEmbeddings(BaseModel, Embeddings):
    """Custom Sagemaker Inference Endpoints.

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

            from langchain_community.embeddings import SagemakerEndpointEmbeddings
            endpoint_name = (
                "my-endpoint-name"
            )
            region_name = (
                "us-west-2"
            )
            credentials_profile_name = (
                "default"
            )
            se = SagemakerEndpointEmbeddings(
                endpoint_name=endpoint_name,
                region_name=region_name,
                credentials_profile_name=credentials_profile_name
            )

            #Use with boto3 client
            client = boto3.client(
                        "sagemaker-runtime",
                        region_name=region_name
                    )
            se = SagemakerEndpointEmbeddings(
                endpoint_name=endpoint_name,
                client=client
            )
    """
    client: Any = None

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

    content_handler: EmbeddingsContentHandler
    """The content handler class that provides an input and
    output transform functions to handle formats between LLM
    and the endpoint.
    """

    """
     Example:
        .. code-block:: python

        from langchain_community.embeddings.sagemaker_endpoint import EmbeddingsContentHandler

        class ContentHandler(EmbeddingsContentHandler):
                content_type = "application/json"
                accepts = "application/json"

                def transform_input(self, prompts: List[str], model_kwargs: Dict) -> bytes:
                    input_str = json.dumps({prompts: prompts, **model_kwargs})
                    return input_str.encode('utf-8')

                def transform_output(self, output: bytes) -> List[List[float]]:
                    response_json = json.loads(output.read().decode("utf-8"))
                    return response_json["vectors"]
    """  # noqa: E501

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

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Dont do anything if client provided externally"""
        if values.get("client") is not None:
            return values

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
                    f"profile name are valid. {e}"
                ) from e

        except ImportError:
            raise ImportError(
                "Could not import boto3 python package. "
                "Please install it with `pip install boto3`."
            )
        return values

    def _embedding_func(self, texts: List[str]) -> List[List[float]]:
        """Call out to SageMaker Inference embedding endpoint."""
        # replace newlines, which can negatively affect performance.
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        _model_kwargs = self.model_kwargs or {}
        _endpoint_kwargs = self.endpoint_kwargs or {}

        body = self.content_handler.transform_input(texts, _model_kwargs)
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

    def embed_documents(
        self, texts: List[str], chunk_size: int = 64
    ) -> List[List[float]]:
        """Compute doc embeddings using a SageMaker Inference Endpoint.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size defines how many input texts will
                be grouped together as request. If None, will use the
                chunk size specified by the class.


        Returns:
            List of embeddings, one for each text.
        """
        results = []
        _chunk_size = len(texts) if chunk_size > len(texts) else chunk_size
        for i in range(0, len(texts), _chunk_size):
            response = self._embedding_func(texts[i : i + _chunk_size])
            results.extend(response)
        return results

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a SageMaker inference endpoint.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self._embedding_func([text])[0]
