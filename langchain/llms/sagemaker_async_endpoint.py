from abc import abstractmethod
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.utils import enforce_stop_tokens
from langchain.llms.sagemaker_endpoint import SagemakerEndpoint
import boto3
import time
import os
import uuid
from botocore.exceptions import ClientError


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
        s3_client=None,
        max_retries: int = 25,
        retry_delay: int = 5
    ):
        """
        Wait for the inference file to be generated in S3.
        Args:
            output_url: The S3 URL of the output file.
            failure_url: The S3 URL of the failure file.
            s3_client: The S3 client to use. If None, a new client will be created.
            max_retries: The maximum number of retries to wait for the file.
            retry_delay: The delay in seconds between retries.
        Raises:
            Exception: If the file is not generated within the maximum retries.
        """
        s3_client = boto3.client("s3") if s3_client is None else s3_client
        bucket = output_url.split("/")[2]
        output_prefix = "/".join(output_url.split("/")[3:])
        failure_prefix = "/".join(failure_url.split("/")[3:])

        for retry in range(max_retries):
            try:
                # Check if output_url exists
                response = s3_client.get_object(Bucket=bucket, Key=output_prefix)
                print(response)
                return response
            except ClientError as ex:
                if ex.response['Error']['Code'] == 'NoSuchKey':
                    try:
                        response = s3_client.get_object(Bucket=bucket, Key=failure_prefix)
                        raise Exception(response['Body'].read().decode('utf-8'))
                    except ClientError as ex:
                        if ex.response['Error']['Code'] == 'NoSuchKey':
                            # Wait for file to be generated
                            print("Waiting for file to be generated...")
                            time.sleep(retry_delay)
                        else:
                            raise
                else:
                    raise
        else:
            raise Exception("Exceeded maximum retries while waiting for file to be generated.")

    def __init__(
        self,
        input_bucket: str = "",
        input_prefix: str = "",
        max_request_timeout: int = 90,
        **kwargs
    ):
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
        # Validate input_bucket argument
        if not isinstance(input_bucket, str):
            raise ValueError("input_bucket must be a string.")

        # Validate input_prefix argument
        if not isinstance(input_prefix, str):
            raise ValueError("input_prefix must be a string.")

        # Validate max_request_timeout argument
        if not isinstance(max_request_timeout, int):
            raise ValueError("max_request_timeout must be an integer.")
        if max_request_timeout <= 0:
            raise ValueError("max_request_timeout must be a positive integer.")

        super().__init__(**kwargs)
        region = self.region_name
        account = boto3.client("sts").get_caller_identity()["Account"]
        self.input_bucket = f"sagemaker-{region}-{account}" if input_bucket == "" else input_bucket
        self.input_prefix = f"async-endpoint-outputs/{self.endpoint_name}" if input_prefix == "" else input_prefix
        self.max_request_timeout = max_request_timeout
        self.s3_client = boto3.client("s3")
        self.sm_client = boto3.client("sagemaker")

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
        response = self.sm_client.describe_endpoint(EndpointName=self.endpoint_name)
        endpoint_is_running = response["ProductionVariants"][0]["CurrentInstanceCount"] > 0

        # If the endpoint is not running, send an empty request to "wake up" the endpoint
        test_data = b""
        test_key = os.path.join(self.input_prefix, "test")
        self.s3_client.put_object(Body=test_data, Bucket=self.input_bucket, Key=test_key)
        if not endpoint_is_running:
            self.client.invoke_endpoint_async(
                EndpointName=self.endpoint_name,
                InputLocation=f"s3://{self.input_bucket}/{test_key}",
                ContentType=content_type,
                Accept=accepts,
                InvocationTimeoutSeconds=self.max_request_timeout,  # timeout of 60 seconds to detect if it's not running yet
                **_endpoint_kwargs,
            )
            raise Exception("The endpoint is not running. Please check back in approximately 10 minutes.")
        else:
            print("Endpoint is running! Proceeding to inference.")

        # Send request to the async endpoint
        request_key = os.path.join(self.input_prefix, f"request-{str(uuid.uuid4())}")
        self.s3_client.put_object(Body=body, Bucket=self.input_bucket, Key=request_key)
        response = self.client.invoke_endpoint_async(
            EndpointName=self.endpoint_name,
            InputLocation=f"s3://{self.input_bucket}/{request_key}",
            ContentType=content_type,
            Accept=accepts,
            InvocationTimeoutSeconds=self.max_request_timeout,
            **_endpoint_kwargs,
        )

        # Read the bytes of the file from S3 in output_url with Boto3
        output_url = response["OutputLocation"]
        failure_url = response["FailureLocation"]
        response = self.wait_inference_file(output_url, failure_url, self.s3_client)
        text = self.content_handler.transform_output(response["Body"])
        if stop is not None:
            text = enforce_stop_tokens(text, stop)

        return text
