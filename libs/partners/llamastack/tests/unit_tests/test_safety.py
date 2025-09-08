"""Unit tests for LlamaStack safety module."""

import os
from typing import Any
from unittest.mock import Mock, patch

from langchain_llamastack.safety import LlamaStackSafety, SafetyResult


class TestSafetyResult:
    """Test cases for SafetyResult model."""

    def test_safety_result_creation(self) -> None:
        """Test SafetyResult creation with default values."""
        result = SafetyResult(is_safe=True)

        assert result.is_safe is True
        assert result.violations == []
        assert result.confidence_score is None
        assert result.explanation is None
        assert result.metadata is None

    def test_safety_result_creation_with_values(self) -> None:
        """Test SafetyResult creation with all values."""
        violations = [{"category": "hate", "score": 0.9}]
        result = SafetyResult(
            is_safe=False,
            violations=violations,
            confidence_score=0.95,
            explanation="Content contains hate speech",
            metadata={"model": "test"},
        )

        assert result.is_safe is False
        assert result.violations == violations
        assert result.confidence_score == 0.95
        assert result.explanation == "Content contains hate speech"
        assert result.metadata == {"model": "test"}


class TestLlamaStackSafety:
    """Test cases for LlamaStackSafety."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.base_url = "http://test-server:8321"
        self.shield_type = "llama_guard"

    @patch("langchain_llamastack.safety.LlamaStackClient", None)
    @patch("langchain_llamastack.safety.AsyncLlamaStackClient", None)
    def test_init_default_params(self) -> None:
        """Test initialization with default parameters."""
        with patch.dict(os.environ, {}, clear=True):
            safety = LlamaStackSafety()

            assert safety.base_url == "http://localhost:8321"
            assert safety.shield_type == "llama_guard"
            assert safety.timeout == 30.0
            assert safety.max_retries == 2
            assert safety.client is None
            assert safety.async_client is None

    @patch("langchain_llamastack.safety.LlamaStackClient", None)
    @patch("langchain_llamastack.safety.AsyncLlamaStackClient", None)
    def test_init_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        safety = LlamaStackSafety(
            base_url=self.base_url,
            shield_type=self.shield_type,
            timeout=60.0,
            max_retries=5,
        )

        assert safety.base_url == self.base_url
        assert safety.shield_type == self.shield_type
        assert safety.timeout == 60.0
        assert safety.max_retries == 5

    @patch("langchain_llamastack.safety.LlamaStackClient", None)
    @patch("langchain_llamastack.safety.AsyncLlamaStackClient", None)
    def test_init_env_vars(self) -> None:
        """Test initialization with environment variables."""
        env_vars = {"LLAMA_STACK_BASE_URL": "http://env-server:8321"}

        with patch.dict(os.environ, env_vars):
            safety = LlamaStackSafety()

            assert safety.base_url == "http://env-server:8321"

    @patch("langchain_llamastack.safety.LlamaStackClient")
    def test_get_client_kwargs(self, mock_client_class: Any) -> None:
        """Test getting client kwargs."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        safety = LlamaStackSafety(base_url=self.base_url, timeout=60.0, max_retries=5)

        kwargs = safety._get_client_kwargs()

        expected = {"base_url": self.base_url, "timeout": 60.0, "max_retries": 5}
        assert kwargs == expected

    @patch("langchain_llamastack.safety.LlamaStackClient")
    def test_initialize_client(self, mock_client_class: Any) -> None:
        """Test initializing client."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        safety = LlamaStackSafety(base_url=self.base_url)
        safety._initialize_client()

        assert safety.client == mock_client
        mock_client_class.assert_called_once_with(
            base_url=self.base_url, timeout=30.0, max_retries=2
        )

    @patch("langchain_llamastack.safety.AsyncLlamaStackClient")
    def test_initialize_async_client(self, mock_async_client_class: Any) -> None:
        """Test initializing async client."""
        mock_async_client = Mock()
        mock_async_client_class.return_value = mock_async_client

        safety = LlamaStackSafety(base_url=self.base_url)
        safety._initialize_async_client()

        assert safety.async_client == mock_async_client
        mock_async_client_class.assert_called_once_with(
            base_url=self.base_url, timeout=30.0, max_retries=2
        )

    @patch("langchain_llamastack.safety.LlamaStackClient")
    def test_check_content_safety_success(self, mock_client_class: Any) -> None:
        """Test successful content safety check."""
        # Setup mock client and response
        mock_client = Mock()
        mock_response = Mock()

        # Configure mock response to NOT be a violation
        mock_response.is_violation = False
        mock_response.confidence_score = 0.95
        mock_response.explanation = "Content is safe"

        mock_client.safety.run_shield.return_value = mock_response
        mock_client_class.return_value = mock_client

        safety = LlamaStackSafety(base_url=self.base_url)
        result = safety.check_content_safety("Hello world")

        assert isinstance(result, SafetyResult)
        assert result.is_safe is True
        assert result.violations == []
        assert result.confidence_score == 0.95
        assert result.explanation == "Content is safe"

        # Verify API call
        mock_client.safety.run_shield.assert_called_once_with(
            shield_type="llama_guard",
            messages=[{"content": "Hello world", "role": "user"}],
        )

    @patch("langchain_llamastack.safety.LlamaStackClient")
    def test_check_content_safety_violation(self, mock_client_class: Any) -> None:
        """Test content safety check with violation."""
        mock_client = Mock()
        mock_response = Mock()

        # Configure mock response to BE a violation
        mock_response.is_violation = True
        mock_response.violation_level = "high"
        mock_response.metadata = {"category": "hate"}
        mock_response.confidence_score = 0.92

        # Ensure hasattr returns True for these attributes
        def mock_hasattr(obj, attr):
            return attr in [
                "is_violation",
                "violation_level",
                "confidence_score",
                "metadata",
            ]

        import builtins

        original_hasattr = builtins.hasattr
        builtins.hasattr = mock_hasattr

        mock_client.safety.run_shield.return_value = mock_response
        mock_client_class.return_value = mock_client

        safety = LlamaStackSafety(base_url=self.base_url)
        result = safety.check_content_safety("Bad content")

        # Restore original hasattr
        builtins.hasattr = original_hasattr

        assert result.is_safe is False
        assert len(result.violations) == 1
        assert result.violations[0]["category"] == "safety_violation"
        assert result.violations[0]["level"] == "high"
        assert result.violations[0]["metadata"] == {"category": "hate"}
        assert result.confidence_score == 0.92

    @patch("langchain_llamastack.safety.LlamaStackClient")
    def test_check_content_safety_error(self, mock_client_class: Any) -> None:
        """Test content safety check with error."""
        mock_client = Mock()
        mock_client.safety.run_shield.side_effect = Exception("API error")
        mock_client_class.return_value = mock_client

        safety = LlamaStackSafety(base_url=self.base_url)
        result = safety.check_content_safety("Test content")

        assert result.is_safe is True  # Fails open
        assert result.violations == []
        assert "Safety check failed: API error" in result.explanation

    @patch("langchain_llamastack.safety.AsyncLlamaStackClient")
    def test_acheck_content_safety_success(self, mock_async_client_class: Any) -> None:
        """Test successful async content safety check."""
        import asyncio

        # Create properly configured mocks
        mock_async_client = Mock()
        mock_response = Mock()

        # Configure the mock response to simulate a safe result
        mock_response.is_violation = False
        mock_response.confidence_score = 0.98
        mock_response.explanation = "Safe content"

        # Make the async call return a coroutine that resolves to the mock response
        async def mock_run_shield(*args, **kwargs):
            return mock_response

        mock_async_client.safety.run_shield = mock_run_shield
        mock_async_client_class.return_value = mock_async_client

        # Test
        safety = LlamaStackSafety(base_url=self.base_url)

        async def test_async_check():
            return await safety.acheck_content_safety("Hello async world")

        result = asyncio.run(test_async_check())

        assert result.is_safe is True
        assert result.confidence_score == 0.98
        assert result.explanation == "Safe content"

    @patch("langchain_llamastack.safety.AsyncLlamaStackClient")
    def test_acheck_content_safety_violation(
        self, mock_async_client_class: Any
    ) -> None:
        """Test async content safety check with violation."""
        import asyncio

        # Create properly configured mocks
        mock_async_client = Mock()

        # Create mock response with explicit attribute configuration
        mock_response = Mock()
        mock_response.configure_mock(
            **{
                "is_violation": True,
                "violation_level": "medium",
                "metadata": {"reason": "inappropriate"},
                "confidence_score": 0.85,
            }
        )

        # Ensure hasattr works correctly
        def custom_hasattr(obj: Any, name: str) -> bool:
            return name in [
                "is_violation",
                "violation_level",
                "metadata",
                "confidence_score",
            ]

        # Make the async call return a coroutine that resolves to the mock response
        async def mock_run_shield(*args, **kwargs):
            return mock_response

        mock_async_client.safety.run_shield = mock_run_shield
        mock_async_client_class.return_value = mock_async_client

        safety = LlamaStackSafety(base_url=self.base_url)

        # Patch hasattr to work with our mock
        import builtins

        original_hasattr = builtins.hasattr

        def patched_hasattr(obj, name):
            if obj is mock_response:
                return custom_hasattr(obj, name)
            return original_hasattr(obj, name)

        builtins.hasattr = patched_hasattr

        try:

            async def test_async_check():
                return await safety.acheck_content_safety("Bad async content")

            result = asyncio.run(test_async_check())

            assert result.is_safe is False
            assert len(result.violations) == 1
            assert result.violations[0]["category"] == "safety_violation"
            assert result.violations[0]["level"] == "medium"
            assert result.confidence_score == 0.85
        finally:
            # Restore original hasattr
            builtins.hasattr = original_hasattr

    @patch("langchain_llamastack.safety.AsyncLlamaStackClient")
    def test_acheck_content_safety_error(self, mock_async_client_class: Any) -> None:
        """Test async content safety check with error."""
        mock_async_client = Mock()
        mock_async_client.safety.run_shield.side_effect = Exception("Async API error")
        mock_async_client_class.return_value = mock_async_client

        safety = LlamaStackSafety(base_url=self.base_url)

        import asyncio

        async def mock_acheck() -> Any:
            return await safety.acheck_content_safety("Test async content")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(mock_acheck())
        finally:
            loop.close()

        assert result.is_safe is True  # Fails open
        assert "Async safety check failed: Async API error" in result.explanation

    @patch("langchain_llamastack.safety.LlamaStackClient")
    def test_client_initialization_lazy(self, mock_client_class: Any) -> None:
        """Test that clients are initialized lazily."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        safety = LlamaStackSafety(base_url=self.base_url)

        # Client should not be initialized yet
        assert safety.client is None

        # Calling a method should initialize the client
        safety.check_content_safety("test")

        assert safety.client is not None

    def test_content_safety_custom_shield_type(self) -> None:
        """Test content safety check with custom shield type."""
        with patch("langchain_llamastack.safety.LlamaStackClient") as mock_client_class:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.is_violation = False

            mock_client.safety.run_shield.return_value = mock_response
            mock_client_class.return_value = mock_client

            safety = LlamaStackSafety(
                base_url=self.base_url, shield_type="custom_shield"
            )
            safety.check_content_safety("test content")

            mock_client.safety.run_shield.assert_called_with(
                shield_type="custom_shield",
                messages=[{"content": "test content", "role": "user"}],
            )

    def test_check_missing_methods(self) -> None:
        """Test that removed methods do not exist."""
        safety = LlamaStackSafety()

        # These methods should not exist anymore
        assert not hasattr(safety, "moderate_content")
        assert not hasattr(safety, "amoderate_content")

        # Only these methods should exist
        assert hasattr(safety, "check_content_safety")
        assert hasattr(safety, "acheck_content_safety")
