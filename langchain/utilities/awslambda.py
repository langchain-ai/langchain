"""Util that calls Lambda."""
import json
from typing import Any, Dict, Optional

from pydantic import BaseModel, Extra, root_validator


class LambdaWrapper(BaseModel):
    """Wrapper for AWS Lambda SDK.

    Docs for using:

    1. pip install boto3
    2. Create a lambda function using the AWS Console or CLI
    3. Run `aws configure` and enter your AWS credentials

    """

    lambda_client: Any  #: :meta private:
    function_name: Optional[str] = None
    awslambda_tool_name: Optional[str] = None
    awslambda_tool_description: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that python package exists in environment."""

        try:
            import boto3

        except ImportError:
            raise ImportError(
                "boto3 is not installed. Please install it with `pip install boto3`"
            )

        values["lambda_client"] = boto3.client("lambda")
        values["function_name"] = values["function_name"]

        return values

    def run(self, query: str) -> str:
        """Invoke Lambda function and parse result."""
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
