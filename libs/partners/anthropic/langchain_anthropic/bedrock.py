"""Anthropic Bedrock chat models."""

import os
import re
from functools import cached_property
from typing import Any

from langchain_core.language_models.chat_models import LangSmithParams
from langchain_core.utils import secret_from_env
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_anthropic._bedrock_utils import _create_bedrock_client_params
from langchain_anthropic.chat_models import ChatAnthropic, _get_default_model_profile


class ChatAnthropicBedrock(ChatAnthropic):
    """Anthropic Claude via AWS Bedrock.

    Uses the `AnthropicBedrock` clients in the `anthropic` SDK.

    See the [LangChain docs for `ChatAnthropic`](https://docs.langchain.com/oss/python/integrations/chat/anthropic)
    for tutorials, feature walkthroughs, and examples.

    See the [Claude Platform docs](https://platform.claude.com/docs/en/about-claude/models/overview)
    for a list of the latest models, their capabilities, and pricing.

    Example:
        ```python
        # pip install -U langchain-anthropic
        # export AWS_ACCESS_KEY_ID="your-access-key"
        # export AWS_SECRET_ACCESS_KEY="your-secret-key"
        # export AWS_REGION="us-east-1"  # or AWS_DEFAULT_REGION

        from langchain_anthropic import ChatAnthropicBedrock

        model = ChatAnthropicBedrock(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            # region_name="us-east-1",  # optional, inferred from env if not provided
            # temperature=,
            # max_tokens=,
            # thinking={"type": "enabled", "budget_tokens": 5000},
        )
        ```

    Note:
        Any param which is not explicitly supported will be passed directly to
        [`AnthropicBedrock.messages.create(...)`](https://docs.anthropic.com/en/api/messages)
        each time the model is invoked.
    """

    model_config = ConfigDict(
        populate_by_name=True,
    )

    region_name: str | None = None
    """The aws region, e.g., `us-west-2`.

    Falls back to AWS_REGION or AWS_DEFAULT_REGION env variable or region specified in
    ~/.aws/config in case it is not provided here.
    """

    aws_access_key_id: SecretStr | None = Field(
        default_factory=secret_from_env("AWS_ACCESS_KEY_ID", default=None)
    )
    """AWS access key id.

    If provided, aws_secret_access_key must also be provided.
    If not specified, the default credential profile or, if on an EC2 instance,
    credentials from IMDS will be used.
    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If not provided, will be read from 'AWS_ACCESS_KEY_ID' environment variable.

    """

    aws_secret_access_key: SecretStr | None = Field(
        default_factory=secret_from_env("AWS_SECRET_ACCESS_KEY", default=None)
    )
    """AWS secret_access_key.

    If provided, aws_access_key_id must also be provided.
    If not specified, the default credential profile or, if on an EC2 instance,
    credentials from IMDS will be used.
    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If not provided, will be read from 'AWS_SECRET_ACCESS_KEY' environment variable.
    """

    aws_session_token: SecretStr | None = Field(
        default_factory=secret_from_env("AWS_SESSION_TOKEN", default=None)
    )
    """AWS session token.

    If provided, aws_access_key_id and aws_secret_access_key must
    also be provided. Not required unless using temporary credentials.
    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If not provided, will be read from 'AWS_SESSION_TOKEN' environment variable.
    """

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "anthropic-bedrock-chat"

    @property
    def lc_secrets(self) -> dict[str, str]:
        """Return a mapping of secret keys to environment variables."""
        return {
            "aws_access_key_id": "AWS_ACCESS_KEY_ID",
            "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",
            "aws_session_token": "AWS_SESSION_TOKEN",
            "mcp_servers": "ANTHROPIC_MCP_SERVERS",
            "anthropic_api_key": "ANTHROPIC_API_KEY",
        }

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "chat_models", "anthropic-bedrock"]`
        """
        return ["langchain", "chat_models", "anthropic-bedrock"]

    @cached_property
    def _client_params(self) -> dict[str, Any]:
        """Get client parameters for AnthropicBedrock."""
        region_name = (
            self.region_name
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
        )
        if not region_name:
            msg = (
                "AWS region must be specified either via the region_name parameter, "
                "AWS_REGION environment variable, AWS_DEFAULT_REGION environment "
                "variable, or ~/.aws/config"
            )
            raise ValueError(msg)
        return _create_bedrock_client_params(
            region_name=region_name,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
            max_retries=self.max_retries,
            default_headers=self.default_headers,
            timeout=self.default_request_timeout,
        )

    @cached_property
    def _client(self) -> Any:  # type: ignore[type-arg]
        """Get synchronous AnthropicBedrock client."""
        try:
            from anthropic import AnthropicBedrock
        except ImportError as e:
            msg = (
                "AnthropicBedrock client is not available. "
                "Please ensure you have anthropic>=0.78.0 installed. "
                "If using an older version, upgrade with: "
                "pip install --upgrade anthropic"
            )
            raise ImportError(msg) from e

        client_params = self._client_params
        return AnthropicBedrock(**client_params)

    @cached_property
    def _async_client(self) -> Any:  # type: ignore[type-arg]
        """Get asynchronous AnthropicBedrock client."""
        try:
            from anthropic import AsyncAnthropicBedrock
        except ImportError as e:
            msg = (
                "AsyncAnthropicBedrock client is not available. "
                "Please ensure you have anthropic>=0.78.0 installed. "
                "If using an older version, upgrade with: "
                "pip install --upgrade anthropic"
            )
            raise ImportError(msg) from e

        client_params = self._client_params
        return AsyncAnthropicBedrock(**client_params)

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="anthropic-bedrock",
            ls_model_name=params.get("model", self.model),
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_max_tokens := params.get("max_tokens", self.max_tokens):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop", None):
            ls_params["ls_stop"] = ls_stop
        return ls_params

    @model_validator(mode="after")
    def _set_model_profile(self) -> Self:
        """Set model profile if not overridden."""
        if self.profile is None:
            # Strip region prefix (e.g., "us."), provider prefix (e.g., "anthropic."),
            # and version suffix (e.g., "-v1:0")
            model_id = re.sub(r"^[A-Za-z]{2}\.", "", self.model)  # Remove region
            model_id = re.sub(r"^anthropic\.", "", model_id)  # Remove provider
            model_id = re.sub(r"-v\d+:\d+$", "", model_id)  # Remove version suffix
            self.profile = _get_default_model_profile(model_id)
        if (
            self.profile is not None
            and self.betas
            and "context-1m-2025-08-07" in self.betas
        ):
            self.profile["max_input_tokens"] = 1_000_000
        return self
