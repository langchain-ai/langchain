"""0G Compute Network LLM implementation."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import aiohttp
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LangSmithParams
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, LLMResult
from langchain_core.utils import from_env, secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_zerog.broker import ZeroGBroker, OFFICIAL_MODELS

logger = logging.getLogger(__name__)


class ZeroGLLM(BaseLLM):
    """0G Compute Network LLM integration.

    This class provides access to 0G's decentralized LLM services for text completion tasks.
    """

    model: str = Field(description="Name of 0G model to use")
    """Name of the 0G model to use."""

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

    temperature: float = Field(default=0.7, description="Sampling temperature")
    """Sampling temperature between 0 and 2."""

    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens to generate"
    )
    """Maximum number of tokens to generate."""

    top_p: float = Field(default=1.0, description="Nucleus sampling parameter")
    """Nucleus sampling parameter."""

    frequency_penalty: float = Field(default=0.0, description="Frequency penalty")
    """Frequency penalty parameter."""

    presence_penalty: float = Field(default=0.0, description="Presence penalty")
    """Presence penalty parameter."""

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize ZeroGLLM."""
        super().__init__(**kwargs)
        self._broker: Optional[ZeroGBroker] = None

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "zerog-llm"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        """A map of constructor argument names to secret ids."""
        return {"private_key": "ZEROG_PRIVATE_KEY"}

    def _get_ls_params(
        self,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        """Get LangSmith parameters."""
        ls_params = super()._get_ls_params(stop=stop, **kwargs)
        ls_params["ls_provider"] = "zerog"
        return ls_params

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate environment and initialize broker."""
        if not (self.private_key and self.private_key.get_secret_value()):
            msg = "ZEROG_PRIVATE_KEY must be set."
            raise ValueError(msg)

        # Set provider address if not provided
        if not self.provider_address and self.model in OFFICIAL_MODELS:
            self.provider_address = OFFICIAL_MODELS[self.model]["provider_address"]
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

    def _create_completion_request(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Create the request payload for the 0G service."""
        request_data = {
            "prompt": prompt,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }

        # Add any additional kwargs
        request_data.update(kwargs)

        # Remove None values
        return {k: v for k, v in request_data.items() if v is not None}

    async def _make_request(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Make a request to the 0G Compute Network."""
        broker = self._get_broker()

        # Ensure provider is acknowledged
        if self.provider_address:
            await broker.acknowledge_provider(self.provider_address)

        # Get service metadata
        metadata = await broker.get_service_metadata(self.provider_address or "")
        endpoint = metadata["endpoint"]

        # Create request payload
        request_payload = self._create_completion_request(prompt, **kwargs)

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
                f"{endpoint}/completions",
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

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Async generate completions for the given prompts."""
        generations = []

        for prompt in prompts:
            # Override stop sequences if provided
            request_kwargs = kwargs.copy()
            if stop:
                request_kwargs["stop"] = stop

            response_data = await self._make_request(prompt, **request_kwargs)

            prompt_generations = []
            for choice in response_data["choices"]:
                text = choice["text"]

                # Extract usage information if available
                usage = response_data.get("usage", {})
                generation_info = {
                    "finish_reason": choice.get("finish_reason"),
                    "model": response_data.get("model", self.model),
                }

                if usage:
                    generation_info["token_usage"] = {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    }

                generation_info.update({
                    "provider_address": self.provider_address,
                    "verification": OFFICIAL_MODELS.get(self.model, {}).get("verification", "None"),
                })

                prompt_generations.append(Generation(text=text, generation_info=generation_info))

            generations.append(prompt_generations)

        return LLMResult(generations=generations)

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate completions for the given prompts."""
        import asyncio

        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we're already in an event loop, we can't use run_until_complete
            msg = (
                "Cannot call synchronous _generate from within an async context. "
                "Use agenerate instead."
            )
            raise RuntimeError(msg)
        except RuntimeError:
            # No running loop, we can create one
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    msg = (
                        "Cannot call synchronous _generate from within an async context. "
                        "Use agenerate instead."
                    )
                    raise RuntimeError(msg)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self._agenerate(prompts, stop=stop, run_manager=run_manager, **kwargs)
        )
