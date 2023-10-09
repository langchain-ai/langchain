from abc import abstractmethod
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.utils import enforce_stop_tokens
from langchain.llms.sagemaker_endpoint import SagemakerEndpoint
import boto3
import time
import os
import uuid
import datetime
import logging
from botocore.exceptions import WaiterError, ClientError


class SagemakerAsyncEndpoint(SagemakerEndpoint):
    input_bucket: str = ""
    input_prefix: str = ""
    max_request_timeout: int = 90
    s3_client: Any
    sm_client: Any
        
    def wait_inference_file(
        self,
        output_url: str,
        failure_url: str,
        s3_client: Any = None,
        max_retries: int = 25,
        retry_delay: int = 5
    ) -> Any:
        """Wait for an inference output file to become available on S3.
        Args:
            output_url (str): S3 URL of the expected output file
            failure_url (str): S3 URL to check for inference failure file
            s3_client (boto3.Client): S3 client to use 
            max_retries (int): Maximum retries to check for output file
            retry_delay (int): Seconds to wait between retries           
        Raises:
            Exception: If failure file exists    
        """
        s3_client = boto3.client("s3") if s3_client is None else s3_client
        bucket = output_url.split("/")[2]
        output_prefix = "/".join(output_url.split("/")[3:])
        failure_prefix = "/".join(failure_url.split("/")[3:])
        
        tries = 0
        while tries < max_retries:
            try:
                waiter = s3_client.get_waiter('object_exists')
                waiter.wait(Bucket=bucket, Key=output_prefix)
                return
            except WaiterError:
                tries += 1
                print(f"Output file not found yet, waiting {retry_delay} seconds...")
                time.sleep(retry_delay)

        # Output file still not available, check failure file
        waiter = s3_client.get_waiter('object_exists') 
        waiter.wait(Bucket=bucket, Key=failure_prefix)
        
        raise Exception("Inference failed while waiting for file to be generated.")

    def __init__(
        self,
        input_bucket: str = "",
        input_prefix: str = "",
        max_request_timeout: int = 90,
        **kwargs
    ) -> None:
        """
        Initialize a Sagemaker asynchronous endpoint connector in Langchain.
        Args:
            input_bucket: S3 bucket name where input files are stored.
            input_prefix: S3 prefix where input files are stored.
            max_request_timeout: Maximum timeout for the request in seconds - also used to validate if endpoint is in cold start.
            kwargs: Keyword arguments to pass to the SagemakerEndpoint class.
        Raises:
            ValueError: If the input_bucket or input_prefix arguments are not of type str,
                or if the max_request_timeout is not a positive integer.
        """
        super().__init__(**kwargs)
        region = self.region_name
        account = boto3.client("sts").get_caller_identity()["Account"]
        self.input_bucket = f"sagemaker-{region}-{account}" if input_bucket == "" else input_bucket
        self.input_prefix = f"async-endpoint-outputs/{self.endpoint_name}" if input_prefix == "" else input_prefix
        self.max_request_timeout = max_request_timeout
        self.s3_client = boto3.client("s3")
        self.sm_client = boto3.client("sagemaker")

    # Private method to invoke endpoint
    def _invoke_endpoint(
        self, 
        input_key: str,
        content_type: str,
        accepts: str,
        **kwargs
    ) -> Any:
        """Invoke SageMaker endpoint asynchronously.

        Args:
            input_key: S3 key for input data 
            content_type: MIME type for input data
            accepts: Expected response MIME type
            **kwargs: Additional parameters for client.invoke_endpoint_async()

        Returns:
            Response dictionary containing InferenceId
        """
        response = self.client.invoke_endpoint_async(
            EndpointName=self.endpoint_name, 
            InputLocation=f"s3://{self.input_bucket}/{input_key}",
            ContentType=content_type,
            Accept=accepts,
            InvocationTimeoutSeconds=self.max_request_timeout,
            **kwargs
        )
        return response
        
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> str:
        """
        Call out to Sagemaker asynchronous inference endpoint.
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
        logger = logging.getLogger(__name__)
        response = self.sm_client.describe_endpoint(EndpointName=self.endpoint_name)
        endpoint_is_running = response["ProductionVariants"][0]["CurrentInstanceCount"] > 0

        # If the endpoint is not running, send an empty request to "wake up" the endpoint
        test_data = b""
        test_key = os.path.join(self.input_prefix, "test")
        self.s3_client.put_object(Body=test_data, Bucket=self.input_bucket, Key=test_key)
        if not endpoint_is_running:
            response = self._invoke_endpoint(
                self.endpoint_name, 
                test_key, 
                content_type, 
                accepts, 
                self.max_request_timeout,
                **_endpoint_kwargs)
            logger.error("The endpoint is not running. Please check back in approximately 10 minutes.")
            raise Exception("The endpoint is not running. Please check back in approximately 10 minutes.")
        else:
            logger.info("Endpoint is running! Proceeding to inference.")

        # Send request to the async endpoint
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")    # including timestamp to avoid collision in a multi-user scenario
        request_key = os.path.join(
            self.input_prefix, 
            f"request-{timestamp}-{str(uuid.uuid4())}"
        )
        self.s3_client.put_object(Body=body, Bucket=self.input_bucket, Key=request_key)
        response = self._invoke_endpoint(
            self.endpoint_name, 
            request_key, 
            content_type, 
            accepts, 
            self.max_request_timeout,
            **_endpoint_kwargs)

        # Read the bytes of the file from S3 in output_url with Boto3
        output_url = response["OutputLocation"]
        failure_url = response["FailureLocation"]
        response = self.wait_inference_file(output_url, failure_url, self.s3_client)
        text = self.content_handler.transform_output(response["Body"])
        if stop is not None:
            text = enforce_stop_tokens(text, stop)

        return text