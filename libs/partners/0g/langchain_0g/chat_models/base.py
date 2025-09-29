"""0g.ai chat wrapper."""

from __future__ import annotations

from typing import Any, Optional

from a0g import A0G  # type: ignore[import]
from langchain_core.utils.utils import secret_from_env
from langchain_openai import ChatOpenAI
from pydantic import Field, SecretStr, model_validator
from typing_extensions import Self
from web3.types import ENS


class ZGChat(ChatOpenAI):  # type: ignore[override]
    """Wrapper for 0g.ai LLMs with LangChain OpenAI interface compatibility.

    This class allows interacting with a 0g.ai on-chain LLM provider using
    a wallet private key for authentication. It provides both synchronous
    and asynchronous OpenAI-compatible clients while integrating with LangChain.

    Attributes:
        provider (ENS): ENS address of the LLM provider on 0g.ai.
        private_key (Optional[SecretStr]): Wallet private key for authentication.
        zg_client (A0G): 0g.ai client instance.
        svc (Any): Service object representing the LLM.
        client (Any): Synchronous OpenAI-compatible completions client.
        async_client (Any): Asynchronous OpenAI-compatible completions client.

    Notes:
        - The class automatically initializes dummy API keys to satisfy
          Pydantic requirements for OpenAI keys.
        - The `bundle.js` must be included in package data for proper
          deployment in wheel distributions (if using CI/CD pipelines).
    """

    provider: ENS = Field(
        default=ENS("0xf07240Efa67755B5311bc75784a061eDB47165Dd"), exclude=True
    )
    """Provider address of LLM in 0g.ai"""

    private_key: Optional[SecretStr] = Field(
        default_factory=secret_from_env("A0G_PRIVATE_KEY", default=None)
    )
    """Private key of wallet in 0g.ai"""

    zg_client: A0G = Field(default=None, exclude=True)
    """A0G client"""

    svc: Any = Field(default=None, exclude=True)
    """Service object"""

    @model_validator(mode="before")
    @classmethod
    def init_api_key(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Suppress api key exception."""
        values = dict(values)
        values["api_key"] = "DUMMY"
        values["openai_api_key"] = SecretStr("DUMMY")  # чтобы Pydantic не ругался
        return values

    @model_validator(mode="after")
    def init_custom_clients(self) -> Self:
        """Initialize zg-client."""
        if self.private_key is None:
            raise
        self.zg_client = A0G(private_key=self.private_key.get_secret_value())
        self.svc = self.zg_client.get_service(self.provider)
        self.root_client = self.zg_client.get_openai_client(self.provider)
        self.root_async_client = self.zg_client.get_openai_async_client(self.provider)
        self.client = self.root_client.chat.completions
        self.async_client = self.root_async_client.chat.completions
        self.model_name = self.svc.model
        self.openai_api_key = SecretStr("")
        return self

    @property
    def lc_secrets(self) -> dict[str, str]:
        """A map of constructor argument names to secret ids.

        For example,
            {"openai_api_key": "OPENAI_API_KEY"}
        """
        return {"private_key": "A0G_PRIVATE_KEY"}

    @property
    def lc_attributes(self) -> dict[str, Any]:
        """List of attribute names that should be included in the serialized kwargs.

        These attributes must be accepted by the constructor.
        Default is an empty dictionary.
        """
        return {}
