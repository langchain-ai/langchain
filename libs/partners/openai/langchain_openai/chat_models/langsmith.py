"""LangSmith model gateway chat model."""

from typing import Any, ClassVar

from langchain_core.utils._gateway import _apply_gateway_config
from openai import OpenAIError
from pydantic import model_validator
from typing_extensions import override

from langchain_openai.chat_models.base import ChatOpenAI

_LANGSMITH_GATEWAY_DEFAULT_BASE = "https://gateway.smith.langchain.com/v1"


class ChatLangSmithGateway(ChatOpenAI):
    """Chat model routed through the LangSmith model gateway.

    The gateway uses the OpenAI-compatible Responses API. Credentials are read
    from `LANGSMITH_GATEWAY_API_KEY`, with `LANGSMITH_API_KEY` as a fallback.
    Gateway configuration is isolated from OpenAI-specific environment variables.
    """

    _api_key_secret_env: ClassVar[str] = "LANGSMITH_GATEWAY_API_KEY"  # noqa: S105

    @model_validator(mode="before")
    @classmethod
    def _configure_langsmith_gateway(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Resolve gateway configuration before OpenAI client initialization."""
        config = _apply_gateway_config(
            values,
            cls,
            base_url_field="openai_api_base",
            api_key_field="openai_api_key",
            provider_path="v1",
            api_key_env=("LANGSMITH_GATEWAY_API_KEY", "LANGSMITH_API_KEY"),
            default_base_url=_LANGSMITH_GATEWAY_DEFAULT_BASE,
        )
        if config.api_key is None:
            msg = (
                "Missing credentials. Set LANGSMITH_GATEWAY_API_KEY or "
                "LANGSMITH_API_KEY, or pass api_key explicitly."
            )
            raise OpenAIError(msg)
        values["use_responses_api"] = True
        return values

    @property
    @override
    def _uses_gateway(self) -> bool:
        """Return `True` because all requests use the LangSmith gateway."""
        return True

    @property
    @override
    def lc_secrets(self) -> dict[str, str]:
        """Map the API key to the gateway-specific environment variable."""
        return {"openai_api_key": "LANGSMITH_GATEWAY_API_KEY"}

    @classmethod
    @override
    def get_lc_namespace(cls) -> list[str]:
        """Get the stable serialization namespace for LangSmith chat models."""
        return ["langchain", "chat_models", "openai"]
