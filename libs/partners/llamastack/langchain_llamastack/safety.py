"""LlamaStack Safety integration for LangChain."""

import logging
import os
from typing import Any, Optional

import requests

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

try:
    from llama_stack_client import (  # type: ignore
        AsyncLlamaStackClient,
        LlamaStackClient,
    )
except ImportError:
    AsyncLlamaStackClient = None  # type: ignore
    LlamaStackClient = None  # type: ignore

from pydantic import BaseModel


class SafetyResult(BaseModel):
    """Result from safety check."""

    is_safe: bool
    violations: list[dict[str, Any]] = []
    confidence_score: Optional[float] = None
    explanation: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class LlamaStackSafety:
    """
    Llama Stack safety and moderation integration.

    Key init args — safety params:
        shield_type: str
            Name of safety shield to use for safety and moderation\
             (default: "llama_guard").

    Key init args — client params:
        base_url: str
            Llama Stack server URL (default: "http://localhost:8321").
        timeout: Optional[float]
            Timeout for requests (default: 30.0).
        max_retries: int
            Max number of retries (default: 2).

    Instantiate:
        .. code-block:: python

            from langchain_llama_stack import LlamaStackSafety

            # For local Llama Stack (no API key needed)
            safety = LlamaStackSafety(
                base_url="http://localhost:8321", shield_type="llama_guard"
            )

            # For remote Llama Stack (with API key)
            safety = LlamaStackSafety(
                base_url="http://remote-llama-stack:8321",
                shield_type="llama_guard",
            )

    Check content safety:
        .. code-block:: python

            result = safety.check_content_safety("This is some text to check")
            print(result.is_safe)
            print(result.violations)

        .. code-block:: python

            SafetyResult(is_safe=True, violations=[], confidence_score=0.95)

    Async:
        .. code-block:: python

            result = await safety.acheck_content_safety("Text to check")

        .. code-block:: python

            SafetyResult(is_safe=True, violations=[])
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        shield_type: str = "llama-guard",
        timeout: Optional[float] = 30.0,
        max_retries: int = 2,
    ):
        """Initialize LlamaStackSafety."""
        self.base_url = base_url or os.environ.get(
            "LLAMA_STACK_BASE_URL", "http://localhost:8321"
        )
        self.shield_type = shield_type
        self.timeout = timeout
        self.max_retries = max_retries

        # Clients will be initialized lazily when needed
        self.client: Optional[LlamaStackClient] = None
        self.async_client: Optional[AsyncLlamaStackClient] = None

    def _get_client_kwargs(self) -> dict[str, Any]:
        """Get common client kwargs."""
        client_kwargs = {
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        return client_kwargs

    def _initialize_client(self) -> None:
        """Initialize the Llama Stack client."""
        if self.client is None and LlamaStackClient is not None:
            self.client = LlamaStackClient(**self._get_client_kwargs())

    def _initialize_async_client(self) -> None:
        """Initialize the async Llama Stack client."""
        if self.async_client is None and AsyncLlamaStackClient is not None:
            self.async_client = AsyncLlamaStackClient(**self._get_client_kwargs())

    def list_shields(self) -> list[str]:
        """List all available shields from LlamaStack server."""
        try:
            if self.client is None:
                self._initialize_client()

            if self.client is None:
                return []

            # Get shields
            shields_response = self.client.shields.list()
            # Handle both list and object with .data attribute
            if hasattr(shields_response, "data"):
                shields_data = shields_response.data
            else:
                shields_data = shields_response

            # Return just the identifiers for easy checking
            shield_ids = [shield.identifier for shield in shields_data]
            return shield_ids

        except Exception as e:
            logger.error(f"Error listing shields: {e}")
            return []

    def check_content_safety(
        self, content: str, content_type: str = "text", **kwargs: Any
    ) -> SafetyResult:
        """
        Check content safety using Llama Stack shields.

        Args:
            content: The content to check for safety
            content_type: Type of content (default: "text")
            **kwargs: Additional parameters for safety checking

        Returns:
            SafetyResult with safety assessment
        """
        # logger.info(f"Starting safety check for content: '{content[:50]}...'")
        # logger.info(f"Using shield_type: {self.shield_type}")
        # logger.info(f"Base URL: {self.base_url}")

        # Check if LlamaStackClient is available
        if LlamaStackClient is None:
            logger.error("LlamaStackClient is None - client not available")
            return SafetyResult(
                is_safe=True,
                violations=[],
                explanation="LlamaStackClient not available - install llama-stack-client",
            )

        logger.info("LlamaStackClient is available")

        if self.client is None:
            logger.info("Client is None, initializing...")
            self._initialize_client()

        try:
            # Ensure client is not None after initialization
            if self.client is None:
                logger.error("Client is still None after initialization")
                return SafetyResult(
                    is_safe=True,
                    violations=[],
                    explanation="LlamaStack client not initialized",
                )

            logger.info("Client initialized successfully")
            logger.info(f"Making API call with shield_id: {self.shield_type}")

            # Use the safety.run_shield method
            response = self.client.safety.run_shield(
                shield_id=self.shield_type,
                messages=[{"content": content, "role": "user"}],
                params={},  # Required empty params dict
                **kwargs,
            )

            # Parse safety response
            is_safe = True
            violations = []
            confidence_score = None
            explanation = None

            # Extract confidence score and explanation if available
            if hasattr(response, "confidence_score"):
                confidence_score = response.confidence_score

            if hasattr(response, "explanation"):
                explanation = response.explanation

            # Check if response indicates a violation
            if response.violation:
                is_safe = False

                # Handle violation metadata - it might be dict or object
                violation_metadata = response.violation.metadata
                if isinstance(violation_metadata, dict):
                    violation_type = violation_metadata.get("violation_type", None)
                    violation_level = violation_metadata.get(
                        "violation_level", "unknown"
                    )
                else:
                    violation_type = getattr(violation_metadata, "violation_type", None)
                    violation_level = getattr(
                        violation_metadata, "violation_level", "unknown"
                    )

                violations.append(
                    {
                        "category": violation_type,
                        "level": violation_level,
                        "metadata": violation_metadata,
                    }
                )

            # logger.info(f"Final result - is_safe: {is_safe}, violations: {violations}")

            return SafetyResult(
                is_safe=is_safe,
                violations=violations,
                confidence_score=confidence_score,
                explanation=explanation,
            )
        except Exception as e:
            logger.error(f"Exception occurred during safety check: {str(e)}")
            logger.error(f"Exception type: {type(e)}")
            # Return safe by default on error, but log the issue
            return SafetyResult(
                is_safe=True,
                violations=[],
                explanation=f"Safety check failed: {str(e)}",
            )

    async def acheck_content_safety(
        self, content: str, content_type: str = "text", **kwargs: Any
    ) -> SafetyResult:
        """
        Async check content safety using Llama Stack safety shields.

        Args:
            content: The content to check for safety
            content_type: Type of content (text, image, etc.)
            **kwargs: Additional parameters for safety check

        Returns:
            SafetyResult with safety assessment
        """
        if not self.async_client:
            self._initialize_async_client()

        try:
            # Ensure async client is not None
            if self.async_client is None:
                raise ValueError("LlamaStack async client not initialized")

            # Use the AsyncLlamaStackClient.safety.run_shield method
            response = await self.async_client.safety.run_shield(
                shield_id=self.shield_type,
                messages=[{"content": content, "role": "user"}],
                params={},  # Required empty params dict
                **kwargs,
            )

            # Parse the response (same logic as sync version)
            is_safe = True
            violations = []
            confidence_score = None
            explanation = None

            # Extract confidence score and explanation if available
            if hasattr(response, "confidence_score"):
                confidence_score = response.confidence_score

            if hasattr(response, "explanation"):
                explanation = response.explanation

            # Check if response indicates a violation
            if response.violation:
                is_safe = False

                # Handle violation metadata - it might be dict or object
                violation_metadata = response.violation.metadata
                if isinstance(violation_metadata, dict):
                    violation_type = violation_metadata.get("violation_type", None)
                    violation_level = violation_metadata.get(
                        "violation_level", "unknown"
                    )
                else:
                    violation_type = getattr(violation_metadata, "violation_type", None)
                    violation_level = getattr(
                        violation_metadata, "violation_level", "unknown"
                    )

                violations.append(
                    {
                        "category": violation_type,
                        "level": violation_level,
                        "metadata": violation_metadata,
                    }
                )

            return SafetyResult(
                is_safe=is_safe,
                violations=violations,
                confidence_score=confidence_score,
                explanation=explanation,
            )
        except Exception as e:
            # Return safe by default on error, but log the issue
            return SafetyResult(
                is_safe=True,
                violations=[],
                explanation=f"Async safety check failed: {str(e)}",
            )
