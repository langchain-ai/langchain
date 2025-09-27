"""0G Compute Network embeddings implementation."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import aiohttp
from langchain_core.embeddings import Embeddings
from langchain_core.utils import from_env, secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_zerog.broker import ZeroGBroker

logger = logging.getLogger(__name__)

# Embedding models that might be available on 0G
EMBEDDING_MODELS = {
    "text-embedding-ada-002": {
        "provider_address": "0x1234567890123456789012345678901234567890",  # Placeholder
        "description": "OpenAI-compatible text embedding model",
        "dimensions": 1536,
    },
    "all-MiniLM-L6-v2": {
        "provider_address": "0x2345678901234567890123456789012345678901",  # Placeholder
        "description": "Sentence transformer embedding model",
        "dimensions": 384,
    },
}


class ZeroGEmbeddings(BaseModel, Embeddings):
    """0G Compute Network embeddings integration.

    This class provides access to 0G's decentralized embedding services.
    """

    model: str = Field(
        default="text-embedding-ada-002",
        description="Name of 0G embedding model to use"
    )
    """Name of the 0G embedding model to use."""

    provider_address: Optional[str] = Field(
        default=None,
        description="Specific provider address to use"
    )
    """Provider address. If not provided, uses official provider for the model."""

    private_key: Optional[SecretStr] = Field(
        default_factory=secret_from_env("ZEROG_PRIVATE_KEY", default=None),
        description="Ethereum private key for wallet authentication"
    )
    """Ethereum private key for wallet authentication."""

    rpc_url: str = Field(
        default_factory=from_env("ZEROG_RPC_URL", default="https://evmrpc-testnet.0g.ai"),
        description="0G Network RPC URL"
    )
    """0G Network RPC URL."""

    broker_url: str = Field(
        default_factory=from_env("ZEROG_BROKER_URL", default="https://broker.0g.ai"),
        description="0G broker service URL"
    )
    """0G broker service URL."""

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize ZeroGEmbeddings."""
        super().__init__(**kwargs)
        self._broker: Optional[ZeroGBroker] = None

    @property
    def lc_secrets(self) -> Dict[str, str]:
        """A map of constructor argument names to secret ids."""
        return {"private_key": "ZEROG_PRIVATE_KEY"}

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate environment and initialize broker."""
        if not (self.private_key and self.private_key.get_secret_value()):
            msg = "ZEROG_PRIVATE_KEY must be set."
            raise ValueError(msg)

        # Set provider address if not provided
        if not self.provider_address and self.model in EMBEDDING_MODELS:
            self.provider_address = EMBEDDING_MODELS[self.model]["provider_address"]
        elif not self.provider_address:
            msg = f"Provider address not found for model {self.model}. Please specify provider_address."
            raise ValueError(msg)

        return self

    def _get_broker(self) -> ZeroGBroker:
        """Get or create the 0G broker instance."""
        if self._broker is None:
            if not self.private_key:
                msg = "Private key is required"
                raise ValueError(msg)
            self._broker = ZeroGBroker(
                private_key=self.private_key.get_secret_value(),
                rpc_url=self.rpc_url,
                broker_url=self.broker_url,
            )
        return self._broker

    async def fund_account(self, amount: str) -> Dict[str, Any]:
        """Add funds to the account."""
        broker = self._get_broker()
        return await broker.fund_account(amount)

    async def get_balance(self) -> Dict[str, str]:
        """Get account balance information."""
        broker = self._get_broker()
        return await broker.get_balance()

    def _create_embedding_request(self, texts: List[str]) -> Dict[str, Any]:
        """Create the request payload for the 0G embedding service."""
        return {
            "input": texts,
            "model": self.model,
        }

    async def _make_request(self, texts: List[str]) -> Dict[str, Any]:
        """Make a request to the 0G Compute Network for embeddings."""
        broker = self._get_broker()

        # Ensure provider is acknowledged
        if self.provider_address:
            await broker.acknowledge_provider(self.provider_address)

        # Get service metadata
        metadata = await broker.get_service_metadata(self.provider_address or "")
        endpoint = metadata["endpoint"]

        # Create request payload
        request_payload = self._create_embedding_request(texts)

        # Get authenticated headers
        content = json.dumps(request_payload)
        headers = await broker.get_request_headers(
            self.provider_address or "",
            content
        )
        headers["Content-Type"] = "application/json"

        # Make the request
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{endpoint}/embeddings",
                headers=headers,
                data=content,
            ) as response:
                response.raise_for_status()
                response_data = await response.json()

        # Process response for verification
        await broker.process_response(
            self.provider_address or "",
            json.dumps(response_data),
        )

        return response_data

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embed search docs."""
        response_data = await self._make_request(texts)

        embeddings = []
        for item in response_data["data"]:
            embeddings.append(item["embedding"])

        return embeddings

    async def aembed_query(self, text: str) -> List[float]:
        """Async embed query text."""
        embeddings = await self.aembed_documents([text])
        return embeddings[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        import asyncio

        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we're already in an event loop, we can't use run_until_complete
            msg = (
                "Cannot call synchronous embed_documents from within an async context. "
                "Use aembed_documents instead."
            )
            raise RuntimeError(msg)
        except RuntimeError:
            # No running loop, we can create one
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    msg = (
                        "Cannot call synchronous embed_documents from within an async context. "
                        "Use aembed_documents instead."
                    )
                    raise RuntimeError(msg)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.aembed_documents(texts))

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        import asyncio

        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we're already in an event loop, we can't use run_until_complete
            msg = (
                "Cannot call synchronous embed_query from within an async context. "
                "Use aembed_query instead."
            )
            raise RuntimeError(msg)
        except RuntimeError:
            # No running loop, we can create one
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    msg = (
                        "Cannot call synchronous embed_query from within an async context. "
                        "Use aembed_query instead."
                    )
                    raise RuntimeError(msg)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.aembed_query(text))
