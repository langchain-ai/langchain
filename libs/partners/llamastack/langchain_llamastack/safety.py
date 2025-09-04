"""Llama Stack safety and moderation integration."""

import os
from typing import Any, Dict, List, Optional
"""LlamaStack Safety integration for LangChain."""

import os
from typing import Any, Dict, List, Optional

from llama_stack_client import AsyncLlamaStackClient, LlamaStackClient
from pydantic import BaseModel


class SafetyResult(BaseModel):
    """Result from safety check."""

    is_safe: bool
    violations: List[Dict[str, Any]] = []
    confidence_score: Optional[float] = None
    explanation: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LlamaStackSafety:
    """
    Llama Stack safety and moderation integration.

    Setup:
        Install ``langchain-llama-stack`` and optionally set environment variables ``LLAMA_STACK_API_KEY`` or ``LLAMA_STACK_BASE_URL``.

        .. code-block:: bash

            pip install -U langchain-llama-stack
            export LLAMA_STACK_BASE_URL="http://localhost:8321"  # Optional, defaults to localhost
            export LLAMA_STACK_API_KEY="your-api-key"  # Optional, only needed for remote servers

    Key init args — safety params:
        shield_type: str
            Name of safety shield to use (default: "llama_guard").
        moderation_model: Optional[str]
            Model to use for content moderation.

    Key init args — client params:
        base_url: str
            Llama Stack server URL (default: "http://localhost:8321").
        api_key: Optional[str]
            API key for authentication (optional for local servers).
        timeout: Optional[float]
            Timeout for requests (default: 30.0).
        max_retries: int
            Max number of retries (default: 2).

    Instantiate:
        .. code-block:: python

            from langchain_llama_stack import LlamaStackSafety

            # For local Llama Stack (no API key needed)
            safety = LlamaStackSafety(
                base_url="http://localhost:8321",
                shield_type="llama_guard",
            )

            # For remote Llama Stack (with API key)
            safety = LlamaStackSafety(
                base_url="http://remote-llama-stack:8321",
                api_key="your-api-key",
                shield_type="llama_guard",
            )

    Check content safety:
        .. code-block:: python

            result = safety.check_content_safety("This is some text to check")
            print(result.is_safe)
            print(result.violations)

        .. code-block:: python

            SafetyResult(is_safe=True, violations=[], confidence_score=0.95)

    Run moderation:
        .. code-block:: python

            moderation_result = safety.moderate_content(
                content="Content to moderate",
                content_type="text"
            )

        .. code-block:: python

            SafetyResult(is_safe=False, violations=[{'category': 'hate', 'severity': 'high'}])

    Async:
        .. code-block:: python

            result = await safety.acheck_content_safety("Text to check")

            # moderation:
            moderation_result = await safety.amoderate_content("Content to moderate")

        .. code-block:: python

            SafetyResult(is_safe=True, violations=[])
    """

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        shield_type: str = "llama_guard",
        moderation_model: Optional[str] = None,
        timeout: Optional[float] = 30.0,
        max_retries: int = 2,
    ):
        """Initialize LlamaStackSafety."""
        self.base_url = base_url or os.environ.get(
            "LLAMA_STACK_BASE_URL", "http://localhost:8321"
        )
        self.api_key = api_key or os.environ.get("LLAMA_STACK_API_KEY")
        self.shield_type = shield_type
        self.moderation_model = moderation_model
        self.timeout = timeout
        self.max_retries = max_retries

        # Clients will be initialized lazily when needed
        self.client: Optional[LlamaStackClient] = None
        self.async_client: Optional[AsyncLlamaStackClient] = None

    def _get_client_kwargs(self) -> Dict[str, Any]:
        """Get common client kwargs."""
        client_kwargs = {
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        if self.api_key is not None:
            client_kwargs["api_key"] = self.api_key
        return client_kwargs

    def _initialize_client(self):
        """Initialize the Llama Stack client."""
        if self.client is None:
            self.client = LlamaStackClient(**self._get_client_kwargs())

    def _initialize_async_client(self):
        """Initialize the async Llama Stack client."""
        if self.async_client is None:
            self.async_client = AsyncLlamaStackClient(**self._get_client_kwargs())

    def check_content_safety(
        self, content: str, content_type: str = "text", **kwargs
    ) -> SafetyResult:
        """
        Check content safety using Llama Stack safety shields.

        Args:
            content: The content to check for safety
            content_type: Type of content (text, image, etc.)
            **kwargs: Additional parameters for safety check

        Returns:
            SafetyResult with safety assessment
        """
        if not self.client:
            self._initialize_client()

        try:
            # Use the safety.run_shield method
            response = self.client.safety.run_shield(
                shield_type=self.shield_type,
                messages=[{"content": content, "role": "user"}],
                **kwargs,
            )

            # Parse the response based on expected format
            is_safe = True
            violations = []
            confidence_score = None
            explanation = None

            if hasattr(response, "is_violation") and response.is_violation:
                is_safe = False
                if hasattr(response, "violation_level"):
                    violations.append(
                        {
                            "category": "safety_violation",
                            "level": response.violation_level,
                            "metadata": getattr(response, "metadata", {}),
                        }
                    )

            if hasattr(response, "confidence_score"):
                confidence_score = response.confidence_score

            if hasattr(response, "explanation"):
                explanation = response.explanation

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
                explanation=f"Safety check failed: {str(e)}",
            )

    def moderate_content(
        self, content: str, content_type: str = "text", **kwargs
    ) -> SafetyResult:
        """
        Moderate content using Llama Stack moderation.

        Args:
            content: The content to moderate
            content_type: Type of content
            **kwargs: Additional parameters for moderation

        Returns:
            SafetyResult with moderation assessment
        """
        if not self.client:
            self._initialize_client()

        try:
            # Use the moderations.create method
            response = self.client.moderations.create(
                content=content, model=self.moderation_model, **kwargs
            )

            # Parse moderation response
            is_safe = True
            violations = []

            if hasattr(response, "results") and response.results:
                result = (
                    response.results[0]
                    if isinstance(response.results, list)
                    else response.results
                )

                if hasattr(result, "flagged") and result.flagged:
                    is_safe = False

                if hasattr(result, "categories"):
                    for category, flagged in result.categories.items():
                        if flagged:
                            score = None
                            if hasattr(result, "category_scores") and hasattr(
                                result.category_scores, category
                            ):
                                score = getattr(result.category_scores, category)
                            violations.append(
                                {
                                    "category": category,
                                    "flagged": flagged,
                                    "score": score,
                                }
                            )

            return SafetyResult(is_safe=is_safe, violations=violations)

        except Exception as e:
            # Return safe by default on error, but log the issue
            return SafetyResult(
                is_safe=True, violations=[], explanation=f"Moderation failed: {str(e)}"
            )

    async def acheck_content_safety(
        self, content: str, content_type: str = "text", **kwargs
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
            # Use the AsyncLlamaStackClient.safety.run_shield method
            response = await self.async_client.safety.run_shield(
                shield_type=self.shield_type,
                messages=[{"content": content, "role": "user"}],
                **kwargs,
            )

            # Parse the response (same logic as sync version)
            is_safe = True
            violations = []
            confidence_score = None
            explanation = None

            if hasattr(response, "is_violation") and response.is_violation:
                is_safe = False
                if hasattr(response, "violation_level"):
                    violations.append(
                        {
                            "category": "safety_violation",
                            "level": response.violation_level,
                            "metadata": getattr(response, "metadata", {}),
                        }
                    )

            if hasattr(response, "confidence_score"):
                confidence_score = response.confidence_score

            if hasattr(response, "explanation"):
                explanation = response.explanation

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

    async def amoderate_content(
        self, content: str, content_type: str = "text", **kwargs
    ) -> SafetyResult:
        """
        Async moderate content using Llama Stack moderation.

        Args:
            content: The content to moderate
            content_type: Type of content
            **kwargs: Additional parameters for moderation

        Returns:
            SafetyResult with moderation assessment
        """
        if not self.async_client:
            self._initialize_async_client()

        try:
            # Use the AsyncLlamaStackClient.moderations.create method
            response = await self.async_client.moderations.create(
                content=content, model=self.moderation_model, **kwargs
            )

            # Parse moderation response (same logic as sync version)
            is_safe = True
            violations = []

            if hasattr(response, "results") and response.results:
                result = (
                    response.results[0]
                    if isinstance(response.results, list)
                    else response.results
                )

                if hasattr(result, "flagged") and result.flagged:
                    is_safe = False

                if hasattr(result, "categories"):
                    for category, flagged in result.categories.items():
                        if flagged:
                            score = None
                            if hasattr(result, "category_scores") and hasattr(
                                result.category_scores, category
                            ):
                                score = getattr(result.category_scores, category)
                            violations.append(
                                {
                                    "category": category,
                                    "flagged": flagged,
                                    "score": score,
                                }
                            )

            return SafetyResult(is_safe=is_safe, violations=violations)

        except Exception as e:
            # Return safe by default on error, but log the issue
            return SafetyResult(
                is_safe=True,
                violations=[],
                explanation=f"Async moderation failed: {str(e)}",
            )
