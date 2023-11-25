"""Sagemaker InvokeEndpoint API using either Realtime or Async Inference Endpoints."""
import datetime
import io
import json
import logging
import os
import time
import uuid
from abc import abstractmethod
from typing import Any, Dict, Generic, Iterator, List, Mapping, Optional, TypeVar, Union

from langchain_core.pydantic_v1 import Extra, root_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens

logger = logging.getLogger(__file__)

INPUT_TYPE = TypeVar("INPUT_TYPE", bound=Union[str, List[str]])
OUTPUT_TYPE = TypeVar("OUTPUT_TYPE", bound=Union[str, List[List[float]], Iterator])


class LineIterator:
    """
    A helper class for parsing the byte stream input.

    The output of the model will be in the following format:

    b'{"outputs": [" a"]}\n'
    b'{"outputs": [" challenging"]}\n'
    b'{"outputs": [" problem"]}\n'
    ...

    While usually each PayloadPart event from the event stream will
    contain a byte array with a full json, this is not guaranteed
    and some of the json objects may be split acrossPayloadPart events.

    For example:

    {'PayloadPart': {'Bytes': b'{"outputs": '}}
    {'PayloadPart': {'Bytes': b'[" problem"]}\n'}}


    This class accounts for this by concatenating bytes written via the 'write' function
    and then exposing a method which will return lines (ending with a '\n' character)
    within the buffer via the 'scan_lines' function.
    It maintains the position of the last read position to ensure
    that previous bytes are not exposed again.

    For more details see:
    https://aws.amazon.com/blogs/machine-learning/elevating-the-generative-ai-experience-introducing-streaming-support-in-amazon-sagemaker-hosting/
    """

    def __init__(self, stream: Any) -> None:
        self.byte_iterator = iter(stream)
        self.buffer = io.BytesIO()
        self.read_pos = 0

    def __iter__(self) -> "LineIterator":
        return self

    def __next__(self) -> Any:
        while True:
            self.buffer.seek(self.read_pos)
            line = self.buffer.readline()
            if line and line[-1] == ord("\n"):
                self.read_pos += len(line)
                return line[:-1]
            try:
                chunk = next(self.byte_iterator)
            except StopIteration:
                if self.read_pos < self.buffer.getbuffer().nbytes:
                    continue
                raise
            if "PayloadPart" not in chunk:
                # Unknown Event Type
                continue
            self.buffer.seek(0, io.SEEK_END)
            self.buffer.write(chunk["PayloadPart"]["Bytes"])


class ContentHandlerBase(Generic[INPUT_TYPE, OUTPUT_TYPE]):
    """A handler class to transform input from LLM to a
    format that SageMaker endpoint expects.

    Similarly, the class handles transforming output from the
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

    content_type: str = "text/plain"
    """The MIME type of the input data passed to endpoint"""

    accepts: str = "text/plain"
    """The MIME type of the response data returned from endpoint"""

    @abstractmethod
    def transform_input(self, prompt: INPUT_TYPE, model_kwargs: Dict) -> bytes:
        """Transforms the input to a format that model can accept
        as the request Body. Should return bytes or seekable file
        like object in the format specified in the content_type
        request header.
        """

    @abstractmethod
    def transform_output(self, output: bytes) -> OUTPUT_TYPE:
        """Transforms the output from the model to string that
        the LLM class expects.
        """


class LLMContentHandler(ContentHandlerBase[str, str]):
    """Content handler for LLM class."""


class _BaseSagemakerEndpoint(LLM):
    """Base Class for Sagemaker Inference Endpoint models."""

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

    content_handler: LLMContentHandler
    """The content handler class that provides an input and
    output transform functions to handle formats between LLM
    and the endpoint.
    """

    """
     Example:
        .. code-block:: python

        from langchain.llms.sagemaker_endpoint import LLMContentHandler

        class ContentHandler(LLMContentHandler):
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
    """Keyword arguments to pass to the model."""

    endpoint_kwargs: Optional[Dict] = None
    """Optional attributes passed to the invoke_endpoint
    function. See `boto3`_. docs for more info.
    .. _boto3: <https://boto3.amazonaws.com/v1/documentation/api/latest/index.html>
    """

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @staticmethod
    def _validate_boto_session(values: Dict) -> Any:
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
        return session

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

    @abstractmethod
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
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


class SagemakerEndpoint(_BaseSagemakerEndpoint):
    """Sagemaker Inference Endpoint models.

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
    Args:        

        region_name: The aws region e.g., `us-west-2`.
            Fallsback to AWS_DEFAULT_REGION env variable
            or region specified in ~/.aws/config.

        credentials_profile_name: The name of the profile in the ~/.aws/credentials
            or ~/.aws/config files, which has either access keys or role information
            specified. If not specified, the default credential profile or, if on an
            EC2 instance, credentials from IMDS will be used.

        client: boto3 client for Sagemaker Endpoint

        content_handler: Implementation for model specific LLMContentHandler 


    Example:
        .. code-block:: python

            from langchain.llms import SagemakerEndpoint
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

        #Use with boto3 client
            client = boto3.client(
                        "sagemaker-runtime",
                        region_name=region_name
                    )

            se = SagemakerEndpoint(
                endpoint_name=endpoint_name,
                client=client
            )

    """

    _client: Any = None
    """Boto3 client for sagemaker runtime"""

    streaming: bool = False
    """Whether to stream the results."""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Dont do anything if client provided externally"""
        if values.get("_client") is not None:
            return values
        session = cls._validate_boto_session(values)
        if values.get("region_name") == "":
            logger.info("Using sessions default region for sagemaker endpoint.")
            values["region_name"] = session.region_name
        values["_client"] = session.client(
            "sagemaker-runtime", region_name=values["region_name"]
        )
        return values

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
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
        _model_kwargs = {**_model_kwargs, **kwargs}
        _endpoint_kwargs = self.endpoint_kwargs or {}

        body = self.content_handler.transform_input(prompt, _model_kwargs)
        content_type = self.content_handler.content_type
        accepts = self.content_handler.accepts

        if self.streaming and run_manager:
            try:
                resp = self._client.invoke_endpoint_with_response_stream(
                    EndpointName=self.endpoint_name,
                    Body=body,
                    ContentType=self.content_handler.content_type,
                    **_endpoint_kwargs,
                )
                iterator = LineIterator(resp["Body"])
                current_completion: str = ""
                for line in iterator:
                    resp = json.loads(line)
                    resp_output = resp.get("outputs")[0]
                    if stop is not None:
                        # Uses same approach as below
                        resp_output = enforce_stop_tokens(resp_output, stop)
                    current_completion += resp_output
                    run_manager.on_llm_new_token(resp_output)
                return current_completion
            except Exception as e:
                raise ValueError(f"Error raised by streaming inference endpoint: {e}")
        else:
            try:
                response = self._client.invoke_endpoint(
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


class SagemakerAsyncEndpoint(_BaseSagemakerEndpoint):
    """Sagemaker Async Inference Endpoint models.

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
    Args:        

        region_name: The aws region e.g., `us-west-2`.
            Fallsback to AWS_DEFAULT_REGION env variable
            or region specified in ~/.aws/config.

        credentials_profile_name: The name of the profile in the ~/.aws/credentials
            or ~/.aws/config files, which has either access keys or role information
            specified. If not specified, the default credential profile or, if on an
            EC2 instance, credentials from IMDS will be used.

        client: boto3 client for Sagemaker Endpoint

        content_handler: Implementation for model specific LLMContentHandler 


    Example:
        .. code-block:: python

            from langchain.llms import SagemakerAsyncEndpoint
            endpoint_name = (
                "my-endpoint-name"
            )
            region_name = (
                "us-west-2"
            )
            credentials_profile_name = (
                "default"
            )
            se = SagemakerAsyncEndpoint(
                endpoint_name=endpoint_name,
                region_name=region_name,
                credentials_profile_name=credentials_profile_name
            )

            #Use with boto3 client
            client = boto3.client(
                        "sagemaker-runtime",
                        region_name=region_name
                    )

            se = SagemakerAsyncEndpoint(
                endpoint_name=endpoint_name,
                client=client
            )

    """

    input_bucket: str = ""
    """Configured input bucket of endpoint. If not configured uses default."""

    input_prefix: str = ""
    """Configured input prefix of endpoint. If not configured uses default."""

    max_request_timeout: int = 90
    """Time until request timeout for endpoint invocation."""

    session: Optional[Any] = None
    """boto3 session. Defaults to a new session."""

    _s3_client: Any = None
    """boto3 s3 client"""

    _sm_client: Any = None
    """boto3 sm client"""

    _smr_client: Any = None
    """boto3 smr client"""

    _waiter_error: Any = None
    """WaiterError from botocore. Will automatically be imported
    and set during class initialization."""

    max_retries: int = 1
    """Maximum retries during polling of async inference results.
    One try polls every 5 seconds for 20 times.
    See: https://boto3.amazonaws.com/v1/documentation/api/1.9.42/reference/services/s3.html#waiters."""

    wake_up_endpoint: bool = True
    """Whether to wake up the endpoint if it is not running
    by sending the request."""

    wake_up_wait: int = 500
    """If the endpoint is not running wait x seconds for scale up."""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate aws environment (boto3 and access) and set inferred defaults."""
        if values.get("session") is None:
            values["session"] = cls._validate_boto_session(values)
        if values.get("region_name") == "":
            logger.info("Using sessions default region for sagemaker endpoint.")
            values["region_name"] = values["session"].region_name
        values["_s3_client"] = values["session"].client("s3")
        values["_sm_client"] = values["session"].client("sagemaker")
        values["_smr_client"] = values["session"].client(
            "sagemaker-runtime", region_name=values["region_name"]
        )
        # this is not considered best practice, however, other tests in langchain
        # rely on imports from this module. botocore is not present during test
        # execution which means all tests importing this module will
        # fail if this import is not shieled inside a function scope
        from botocore.exceptions import WaiterError

        values["_waiter_error"] = WaiterError

        # Also set defaults based on dynamic values
        if values["input_bucket"] == "" or values["input_prefix"] == "":
            sts_client = values["session"].client("sts")
            account_id = sts_client.get_caller_identity()["Account"]
            if values["input_bucket"] == "":
                values[
                    "input_bucket"
                ] = f"sagemaker-{values['session'].region_name}-{account_id}"
            else:
                if values["input_bucket"].startswith("s3://"):
                    raise ValueError(
                        "Input bucket is not a valid s3 bucket."
                        "Must not start with s3://"
                    )
            if values["input_prefix"] == "":
                values["input_prefix"] = "async-endpoint-outputs/"
                f"{values['endpoint_name']}"
        return values

    def _wait_inference_file(
        self,
        output_url: str,
        failure_url: str,
    ) -> Any:
        """Wait for an inference output file to become available on S3.
        Args:
            output_url (str): S3 URL of the expected output file
            failure_url (str): S3 URL to check for inference failure file
        Raises:
            Exception: If failure file exists
        """
        bucket = output_url.split("/")[2]
        output_prefix = "/".join(output_url.split("/")[3:])
        failure_prefix = "/".join(failure_url.split("/")[3:])

        tries = 0
        while tries < self.max_retries:
            try:
                waiter = self._s3_client.get_waiter("object_exists")
                waiter.wait(Bucket=bucket, Key=output_prefix)
                result = self._s3_client.get_object(Bucket=bucket, Key=output_prefix)
                result["failure"] = False
                return result
            except self._waiter_error:
                tries += 1
                logger.info("Output file not found yet.")

        # Output file still not available, check failure file
        try:
            waiter = self._s3_client.get_waiter("object_exists")
            waiter.wait(Bucket=bucket, Key=failure_prefix)
            result = self._s3_client.get_object(Bucket=bucket, Key=failure_prefix)
            result["failure"] = True
            return result
        except self._waiter_error:
            logger.error("Could also find no error log in failure bucket.")
        raise ValueError(
            "Could not fetch a result or error from the Sagemaker Async Endpoint."
        )

    def _invoke_endpoint(
        self, input_key: str, content_type: str, accepts: str, **kwargs: Dict[str, Any]
    ) -> Dict[str, str]:
        """Invoke the sagemaker async endpoint.

        Args:
            input_key (str): the key if the input object
            content_type (str): the type of the content
            accepts (str): what is accepted (e.g application/json)
            kwargs (Dict[str, Any]): Keyword arguments to pass to the
                invoke_endpoint_async endpoint.

        Returns:
            Dict[str, str]: {'InferenceId': 'string',
                            'OutputLocation': 'string',
                            'FailureLocation': 'string'}
        """
        response = self._smr_client.invoke_endpoint_async(
            EndpointName=self.endpoint_name,
            InputLocation=f"s3://{self.input_bucket}/{input_key}",
            ContentType=content_type,
            Accept=accepts,
            InvocationTimeoutSeconds=self.max_request_timeout,
            **kwargs,
        )
        return response

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Call out to Sagemaker asynchronous inference endpoint.

        Streaming is not supported for async endpoints.

        Args:
            prompt: The prompt to use for the inference.
            stop: The stop tokens to use for the inference.
            run_manager: The run manager to use for the inference.
            kwargs: Keyword arguments to pass to the SagemakerEndpoint class.
        Returns:
            The output from the Sagemaker asynchronous inference endpoint.
        """
        # Parse the SagemakerEndpoint class arguments
        _model_kwargs = self.model_kwargs or {}
        _model_kwargs = {**_model_kwargs, **kwargs}
        _endpoint_kwargs = self.endpoint_kwargs or {}

        # Transform the input to match SageMaker expectations
        body = self.content_handler.transform_input(prompt, _model_kwargs)
        content_type = self.content_handler.content_type
        accepts = self.content_handler.accepts

        # Verify if the endpoint is running
        response = self._sm_client.describe_endpoint(EndpointName=self.endpoint_name)
        endpoint_is_running = (
            response["ProductionVariants"][0]["CurrentInstanceCount"] > 0
        )

        # If the endpoint is not running and no wake up is configured raise error
        if not endpoint_is_running and not self.wake_up_endpoint:
            raise ConnectionError("Endpoint is not running.")

        # Send request to the async endpoint
        now = datetime.datetime.now()
        # including timestamp to avoid collision in a multi-user scenario
        timestamp = now.strftime("%Y%m%d%H%M%S")
        request_key = os.path.join(
            self.input_prefix, f"request-{timestamp}-{str(uuid.uuid4())}"
        )
        self._s3_client.put_object(Body=body, Bucket=self.input_bucket, Key=request_key)
        response = self._invoke_endpoint(
            request_key, content_type, accepts, **_endpoint_kwargs
        )

        # Read the bytes of the file from S3 in output_url with boto3
        output_url = response["OutputLocation"]
        failure_url = response["FailureLocation"]

        if not endpoint_is_running:
            logger.warning("Endpoint need's to scale up. Timeout possible.")
            time.sleep(self.wake_up_wait)

        response = self._wait_inference_file(output_url, failure_url)

        # if failure but info on it give a verbose exception
        if response["failure"]:
            failure_description = response["Body"].read().decode("utf-8")
            raise ValueError(f"Endpoint failed. Reason: {failure_description}")

        text = self.content_handler.transform_output(response["Body"])
        if stop is not None:
            text = enforce_stop_tokens(text, stop)

        return text
