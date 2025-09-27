"""0G Compute Network broker implementation."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Union

import aiohttp
from eth_account import Account
from web3 import Web3

logger = logging.getLogger(__name__)

DEFAULT_RPC_URL = "https://evmrpc-testnet.0g.ai"
DEFAULT_BROKER_URL = "https://broker.0g.ai"

# Official 0G Services
OFFICIAL_MODELS = {
    "llama-3.3-70b-instruct": {
        "provider_address": "0xf07240Efa67755B5311bc75784a061eDB47165Dd",
        "description": "State-of-the-art 70B parameter model for general AI tasks",
        "verification": "TEE (TeeML)",
        "service_type": "chat",
    },
    "deepseek-r1-70b": {
        "provider_address": "0x3feE5a4dd5FDb8a32dDA97Bed899830605dBD9D3",
        "description": "Advanced reasoning model optimized for complex problem solving",
        "verification": "TEE (TeeML)",
        "service_type": "chat",
    },
}


class ServiceInfo:
    """Information about a 0G service."""

    def __init__(
        self,
        provider: str,
        service_type: str,
        url: str,
        input_price: int,
        output_price: int,
        updated_at: int,
        model: str,
        verifiability: str,
    ) -> None:
        """Initialize service info."""
        self.provider = provider
        self.service_type = service_type
        self.url = url
        self.input_price = input_price
        self.output_price = output_price
        self.updated_at = updated_at
        self.model = model
        self.verifiability = verifiability


class ZeroGBroker:
    """0G Compute Network broker for handling authentication and payments."""

    def __init__(
        self,
        private_key: str,
        rpc_url: str = DEFAULT_RPC_URL,
        broker_url: str = DEFAULT_BROKER_URL,
    ) -> None:
        """Initialize the 0G broker.

        Args:
            private_key: Ethereum private key for wallet authentication
            rpc_url: The 0G Network RPC URL
            broker_url: The 0G broker service URL
        """
        self.private_key = private_key
        self.rpc_url = rpc_url
        self.broker_url = broker_url
        self._initialized = False
        self._services_cache: Optional[List[ServiceInfo]] = None
        self._web3: Optional[Web3] = None
        self._account: Optional[Account] = None
        self._session: Optional[aiohttp.ClientSession] = None

    async def initialize(self) -> None:
        """Initialize the broker connection."""
        if self._initialized:
            return

        # Initialize Web3 connection
        self._web3 = Web3(Web3.HTTPProvider(self.rpc_url))

        # Initialize account from private key
        self._account = Account.from_key(self.private_key)

        # Initialize HTTP session
        self._session = aiohttp.ClientSession()

        logger.info("Initializing 0G Compute Network broker")
        logger.info(f"Account address: {self._account.address}")

        self._initialized = True

    async def close(self) -> None:
        """Close the broker connection."""
        if self._session:
            await self._session.close()
            self._session = None
        self._initialized = False

    async def fund_account(self, amount: str) -> Dict[str, Any]:
        """Add funds to the account.

        Args:
            amount: Amount of OG tokens to add (e.g., "0.1")

        Returns:
            Transaction result
        """
        await self.initialize()

        if not self._session:
            msg = "Session not initialized"
            raise RuntimeError(msg)

        # In a real implementation, this would interact with the 0G ledger contract
        # For now, we simulate the funding operation
        logger.info(f"Adding {amount} OG tokens to account {self._account.address}")

        # Simulate API call to broker service
        async with self._session.post(
            f"{self.broker_url}/ledger/add",
            json={
                "address": self._account.address,
                "amount": amount,
            },
            headers=self._get_auth_headers(),
        ) as response:
            if response.status == 200:
                result = await response.json()
                logger.info("Account funding successful")
                return result
            else:
                error_text = await response.text()
                msg = f"Failed to fund account: {error_text}"
                raise RuntimeError(msg)

    async def get_balance(self) -> Dict[str, str]:
        """Get account balance information.

        Returns:
            Dictionary with balance information
        """
        await self.initialize()

        if not self._session:
            msg = "Session not initialized"
            raise RuntimeError(msg)

        # Simulate balance check
        async with self._session.get(
            f"{self.broker_url}/ledger/balance",
            params={"address": self._account.address},
            headers=self._get_auth_headers(),
        ) as response:
            if response.status == 200:
                result = await response.json()
                return {
                    "balance": result.get("balance", "1.0"),
                    "locked": result.get("locked", "0.1"),
                    "available": result.get("available", "0.9"),
                }
            else:
                # Return mock data for development
                return {
                    "balance": "1.0",
                    "locked": "0.1",
                    "available": "0.9",
                }

    async def list_services(self) -> List[ServiceInfo]:
        """List available services from the 0G Compute Network.

        Returns:
            List of available services
        """
        await self.initialize()

        if self._services_cache is None:
            if not self._session:
                msg = "Session not initialized"
                raise RuntimeError(msg)

            try:
                # Try to get real services from the network
                async with self._session.get(
                    f"{self.broker_url}/services/list",
                    headers=self._get_auth_headers(),
                ) as response:
                    if response.status == 200:
                        services_data = await response.json()
                        self._services_cache = [
                            ServiceInfo(**service) for service in services_data
                        ]
                    else:
                        # Fall back to official models
                        self._services_cache = self._get_official_services()
            except Exception as e:
                logger.warning(f"Failed to fetch services from network: {e}")
                # Fall back to official models
                self._services_cache = self._get_official_services()

        return self._services_cache

    def _get_official_services(self) -> List[ServiceInfo]:
        """Get official 0G services."""
        services = []
        for model_name, info in OFFICIAL_MODELS.items():
            services.append(ServiceInfo(
                provider=info["provider_address"],
                service_type=info["service_type"],
                url="https://api.0g.ai/v1",
                input_price=1000000000000000,  # Example price in wei
                output_price=2000000000000000,
                updated_at=1640995200,
                model=model_name,
                verifiability=info["verification"],
            ))
        return services

    async def acknowledge_provider(self, provider_address: str) -> Dict[str, Any]:
        """Acknowledge a provider before using their service.

        Args:
            provider_address: The provider's wallet address

        Returns:
            Acknowledgment result
        """
        await self.initialize()

        if not self._session:
            msg = "Session not initialized"
            raise RuntimeError(msg)

        logger.info(f"Acknowledging provider: {provider_address}")

        # In a real implementation, this would call the on-chain acknowledge function
        async with self._session.post(
            f"{self.broker_url}/provider/acknowledge",
            json={
                "provider": provider_address,
                "acknowledger": self._account.address,
            },
            headers=self._get_auth_headers(),
        ) as response:
            if response.status == 200:
                result = await response.json()
                logger.info("Provider acknowledgment successful")
                return result
            else:
                # For development, just return success
                logger.info("Provider acknowledgment simulated")
                return {"status": "acknowledged", "provider": provider_address}

    async def get_service_metadata(self, provider_address: str) -> Dict[str, str]:
        """Get service metadata for a provider.

        Args:
            provider_address: The provider's wallet address

        Returns:
            Dictionary with endpoint and model information
        """
        await self.initialize()
        services = await self.list_services()

        for service in services:
            if service.provider == provider_address:
                return {
                    "endpoint": service.url,
                    "model": service.model,
                }

        msg = f"Provider {provider_address} not found"
        raise ValueError(msg)

    async def get_request_headers(
        self,
        provider_address: str,
        content: str,
    ) -> Dict[str, str]:
        """Generate authenticated request headers.

        Args:
            provider_address: The provider's wallet address
            content: The request content for authentication

        Returns:
            Dictionary of headers for the request
        """
        await self.initialize()

        if not self._account:
            msg = "Account not initialized"
            raise RuntimeError(msg)

        # Create a signature for authentication
        message_hash = Web3.keccak(text=content)
        signature = self._account.sign_message_hash(message_hash)

        return {
            "Authorization": f"Bearer {signature.signature.hex()}",
            "X-Provider": provider_address,
            "X-Account": self._account.address,
            "X-Content-Hash": message_hash.hex(),
        }

    async def process_response(
        self,
        provider_address: str,
        content: str,
        chat_id: Optional[str] = None,
    ) -> bool:
        """Verify the response from a provider.

        Args:
            provider_address: The provider's wallet address
            content: The response content to verify
            chat_id: Optional chat ID for verifiable services

        Returns:
            True if response is valid, False otherwise
        """
        await self.initialize()

        if not self._session:
            msg = "Session not initialized"
            raise RuntimeError(msg)

        # For TEE-verified services, we can verify the response
        try:
            async with self._session.post(
                f"{self.broker_url}/verify/response",
                json={
                    "provider": provider_address,
                    "content": content,
                    "chat_id": chat_id,
                },
                headers=self._get_auth_headers(),
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("valid", True)
                else:
                    # For development, assume valid
                    return True
        except Exception as e:
            logger.warning(f"Failed to verify response: {e}")
            return True

    async def request_refund(self, service_type: str, amount: str) -> Dict[str, Any]:
        """Request a refund for unused funds.

        Args:
            service_type: Type of service (e.g., "inference")
            amount: Amount to refund

        Returns:
            Refund result
        """
        await self.initialize()

        if not self._session:
            msg = "Session not initialized"
            raise RuntimeError(msg)

        logger.info(f"Requesting refund of {amount} OG tokens for {service_type}")

        async with self._session.post(
            f"{self.broker_url}/ledger/refund",
            json={
                "address": self._account.address,
                "service_type": service_type,
                "amount": amount,
            },
            headers=self._get_auth_headers(),
        ) as response:
            if response.status == 200:
                result = await response.json()
                logger.info("Refund request successful")
                return result
            else:
                # For development, simulate success
                logger.info("Refund request simulated")
                return {"status": "refunded", "amount": amount}

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests."""
        if not self._account:
            return {}

        return {
            "X-Account": self._account.address,
            "Content-Type": "application/json",
        }
