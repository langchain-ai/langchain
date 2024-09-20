"""Util that calls Lambda."""

import json
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, model_validator


class LambdaWrapper(BaseModel):
    """Wrapper for AWS Lambda SDK.
    To use, you should have the ``boto3`` package installed
    and a lambda functions built from the AWS Console or
    CLI. Set up your AWS credentials with ``aws configure``

    Example:
        .. code-block:: bash

            pip install boto3

            aws configure

    """

    lambda_client: Any = None  #: :meta private:
    """The configured boto3 client"""
    function_name: Optional[str] = None
    """The name of your lambda function"""
    awslambda_tool_name: Optional[str] = None
    """If passing to an agent as a tool, the tool name"""
    awslambda_tool_description: Optional[str] = None
    """If passing to an agent as a tool, the description"""

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that python package exists in environment."""

        try:
            import boto3

        except ImportError:
            raise ImportError(
                "boto3 is not installed. Please install it with `pip install boto3`"
            )

        values["lambda_client"] = boto3.client("lambda")
        return values

    def run(self, query: str) -> str:
        """
        Invokes the lambda function and returns the
        result.

        Args:
            query: an input to passed to the lambda
                function as the ``body`` of a JSON
                object.
        """
        res = self.lambda_client.invoke(
            FunctionName=self.function_name,
            InvocationType="RequestResponse",
            Payload=json.dumps({"body": query}),
        )

        try:
            payload_stream = res["Payload"]
            payload_string = payload_stream.read().decode("utf-8")
            answer = json.loads(payload_string)["body"]

        except StopIteration:
            return "Failed to parse response from Lambda"

        if answer is None or answer == "":
            # We don't want to return the assumption alone if answer is empty
            return "Request failed."
        else:
            return f"Result: {answer}"
