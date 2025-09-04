"""Unit tests for LlamaStack safety module."""

import os
from unittest.mock import Mock, patch

from langchain_llamastack.safety import LlamaStackSafety, SafetyResult


class TestSafetyResult:
    """Test cases for SafetyResult model."""

    def test_safety_result_creation(self):
        """Test SafetyResult creation with default values."""
        result = SafetyResult(is_safe=True)

        assert result.is_safe is True
        assert result.violations == []
        assert result.confidence_score is None
        assert result.explanation is None
        assert result.metadata is None

    def test_safety_result_creation_with_values(self):
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

    def setup_method(self):
        """Set up test fixtures."""
        self.base_url = "http://test-server:8321"
        self.api_key = "test-api-key"
        self.shield_type = "llama_guard"

    @patch("langchain_llamastack.safety.LlamaStackClient", None)
    @patch("langchain_llamastack.safety.AsyncLlamaStackClient", None)
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        with patch.dict(os.environ, {}, clear=True):
            safety = LlamaStackSafety()

            assert safety.base_url == "http://localhost:8321"
            assert safety.api_key is None
            assert safety.shield_type == "llama_guard"
            assert safety.moderation_model is None
            assert safety.timeout == 30.0
            assert safety.max_retries == 2
            assert safety.client is None
            assert safety.async_client is None

    @patch("langchain_llamastack.safety.LlamaStackClient", None)
    @patch("langchain_llamastack.safety.AsyncLlamaStackClient", None)
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        safety = LlamaStackSafety(
            base_url=self.base_url,
            api_key=self.api_key,
            shield_type=self.shield_type,
            moderation_model="test-model",
            timeout=60.0,
            max_retries=5,
        )

        assert safety.base_url == self.base_url
        assert safety.api_key == self.api_key
        assert safety.shield_type == self.shield_type
        assert safety.moderation_model == "test-model"
        assert safety.timeout == 60.0
        assert safety.max_retries == 5

    @patch("langchain_llamastack.safety.LlamaStackClient", None)
    @patch("langchain_llamastack.safety.AsyncLlamaStackClient", None)
    def test_init_env_vars(self):
        """Test initialization with environment variables."""
        env_vars = {
            "LLAMA_STACK_BASE_URL": "http://env-server:8321",
            "LLAMA_STACK_API_KEY": "env-api-key",
        }

        with patch.dict(os.environ, env_vars):
            safety = LlamaStackSafety()

            assert safety.base_url == "http://env-server:8321"
            assert safety.api_key == "env-api-key"

    @patch("langchain_llamastack.safety.LlamaStackClient")
    def test_get_client_kwargs(self, mock_client_class):
        """Test getting client kwargs."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        safety = LlamaStackSafety(
            base_url=self.base_url, api_key=self.api_key, timeout=60.0, max_retries=5
        )

        kwargs = safety._get_client_kwargs()

        expected = {
            "base_url": self.base_url,
            "timeout": 60.0,
            "max_retries": 5,
            "api_key": self.api_key,
        }
        assert kwargs == expected

    @patch("langchain_llamastack.safety.LlamaStackClient")
    def test_get_client_kwargs_no_api_key(self, mock_client_class):
        """Test getting client kwargs without API key."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        safety = LlamaStackSafety(base_url=self.base_url)

        kwargs = safety._get_client_kwargs()

        expected = {"base_url": self.base_url, "timeout": 30.0, "max_retries": 2}
        assert kwargs == expected
        assert "api_key" not in kwargs

    @patch("langchain_llamastack.safety.LlamaStackClient")
    def test_initialize_client(self, mock_client_class):
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
    def test_initialize_async_client(self, mock_async_client_class):
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
    def test_check_content_safety_success(self, mock_client_class):
        """Test successful content safety check."""
        # Setup mock client and response
        mock_client = Mock()
        mock_response = Mock()
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

    # @patch("langchain_llamastack.safety.LlamaStackClient")
    # def test_check_content_safety_violation(self, mock_client_class):
    #     """Test content safety check with violation."""
    #     mock_client = Mock()
    #     mock_response = Mock()
    #     mock_response.is_violation = True
    #     mock_response.violation_level = "high"
    #     mock_response.metadata = {"category": "hate"}
    #     mock_response.confidence_score = 0.92

    #     mock_client.safety.run_shield.return_value = mock_response
    #     mock_client_class.return_value = mock_client

    #     safety = LlamaStackSafety(base_url=self.base_url)
    #     result = safety.check_content_safety("Bad content")

    #     assert result.is_safe is False
    #     assert len(result.violations) == 1
    #     assert result.violations[0]["category"] == "safety_violation"
    #     assert result.violations[0]["level"] == "high"
    #     assert result.violations[0]["metadata"] == {"category": "hate"}
    #     assert result.confidence_score == 0.92

    @patch("langchain_llamastack.safety.LlamaStackClient")
    def test_check_content_safety_error(self, mock_client_class):
        """Test content safety check with error."""
        mock_client = Mock()
        mock_client.safety.run_shield.side_effect = Exception("API error")
        mock_client_class.return_value = mock_client

        safety = LlamaStackSafety(base_url=self.base_url)
        result = safety.check_content_safety("Test content")

        assert result.is_safe is True  # Fails open
        assert result.violations == []
        assert "Safety check failed: API error" in result.explanation

    @patch("langchain_llamastack.safety.LlamaStackClient")
    def test_moderate_content_success(self, mock_client_class):
        """Test successful content moderation."""
        mock_client = Mock()
        mock_result = Mock()
        mock_result.flagged = False

        mock_response = Mock()
        mock_response.results = mock_result

        mock_client.moderations.create.return_value = mock_response
        mock_client_class.return_value = mock_client

        safety = LlamaStackSafety(base_url=self.base_url)
        result = safety.moderate_content("Clean content")

        assert result.is_safe is True
        assert result.violations == []

        mock_client.moderations.create.assert_called_once_with(
            content="Clean content", model=None
        )

    @patch("langchain_llamastack.safety.LlamaStackClient")
    def test_moderate_content_violation(self, mock_client_class):
        """Test content moderation with violation."""
        mock_client = Mock()

        # Setup mock result with categories
        mock_categories = Mock()
        mock_categories.items.return_value = [("hate", True), ("violence", False)]

        mock_category_scores = Mock()
        mock_category_scores.hate = 0.85

        mock_result = Mock()
        mock_result.flagged = True
        mock_result.categories = mock_categories
        mock_result.category_scores = mock_category_scores

        mock_response = Mock()
        mock_response.results = mock_result

        mock_client.moderations.create.return_value = mock_response
        mock_client_class.return_value = mock_client

        safety = LlamaStackSafety(base_url=self.base_url)
        result = safety.moderate_content("Hateful content")

        assert result.is_safe is False
        assert len(result.violations) == 1
        assert result.violations[0]["category"] == "hate"
        assert result.violations[0]["flagged"] is True
        assert result.violations[0]["score"] == 0.85

    @patch("langchain_llamastack.safety.LlamaStackClient")
    def test_moderate_content_list_results(self, mock_client_class):
        """Test content moderation with list results."""
        mock_client = Mock()

        mock_result = Mock()
        mock_result.flagged = False

        mock_response = Mock()
        mock_response.results = [mock_result]  # List instead of single result

        mock_client.moderations.create.return_value = mock_response
        mock_client_class.return_value = mock_client

        safety = LlamaStackSafety(base_url=self.base_url)
        result = safety.moderate_content("Test content")

        assert result.is_safe is True

    @patch("langchain_llamastack.safety.LlamaStackClient")
    def test_moderate_content_error(self, mock_client_class):
        """Test content moderation with error."""
        mock_client = Mock()
        mock_client.moderations.create.side_effect = Exception("Moderation error")
        mock_client_class.return_value = mock_client

        safety = LlamaStackSafety(base_url=self.base_url)
        result = safety.moderate_content("Test content")

        assert result.is_safe is True  # Fails open
        assert result.violations == []
        assert "Moderation failed: Moderation error" in result.explanation

    # @patch("langchain_llamastack.safety.AsyncLlamaStackClient")
    # def test_acheck_content_safety_success(self, mock_async_client_class):
    #     """Test successful async content safety check."""
    #     mock_async_client = Mock()
    #     mock_response = Mock()
    #     mock_response.is_violation = False
    #     mock_response.confidence_score = 0.98

    #     mock_async_client.safety.run_shield.return_value = mock_response
    #     mock_async_client_class.return_value = mock_async_client

    #     safety = LlamaStackSafety(base_url=self.base_url)

    #     # Create a mock coroutine
    #     import asyncio

    #     async def mock_acheck():
    #         return await safety.acheck_content_safety("Hello async world")

    #     # Run the async function
    #     loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(loop)
    #     try:
    #         result = loop.run_until_complete(mock_acheck())
    #     finally:
    #         loop.close()

    #     assert result.is_safe is True
    #     assert result.confidence_score == 0.98

    #     mock_async_client.safety.run_shield.assert_called_once_with(
    #         shield_type="llama_guard",
    #         messages=[{"content": "Hello async world", "role": "user"}],
    #     )

    # @patch("langchain_llamastack.safety.AsyncLlamaStackClient")
    # def test_acheck_content_safety_violation(self, mock_async_client_class):
    #     """Test async content safety check with violation."""
    #     mock_async_client = Mock()
    #     mock_response = Mock()
    #     mock_response.is_violation = True
    #     mock_response.violation_level = "medium"
    #     mock_response.metadata = {"reason": "inappropriate"}

    #     mock_async_client.safety.run_shield.return_value = mock_response
    #     mock_async_client_class.return_value = mock_async_client

    #     safety = LlamaStackSafety(base_url=self.base_url)

    #     import asyncio

    #     async def mock_acheck():
    #         return await safety.acheck_content_safety("Bad async content")

    #     loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(loop)
    #     try:
    #         result = loop.run_until_complete(mock_acheck())
    #     finally:
    #         loop.close()

    #     assert result.is_safe is False
    #     assert len(result.violations) == 1
    #     assert result.violations[0]["level"] == "medium"

    @patch("langchain_llamastack.safety.AsyncLlamaStackClient")
    def test_acheck_content_safety_error(self, mock_async_client_class):
        """Test async content safety check with error."""
        mock_async_client = Mock()
        mock_async_client.safety.run_shield.side_effect = Exception("Async API error")
        mock_async_client_class.return_value = mock_async_client

        safety = LlamaStackSafety(base_url=self.base_url)

        import asyncio

        async def mock_acheck():
            return await safety.acheck_content_safety("Test async content")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(mock_acheck())
        finally:
            loop.close()

        assert result.is_safe is True  # Fails open
        assert "Async safety check failed: Async API error" in result.explanation

    @patch("langchain_llamastack.safety.AsyncLlamaStackClient")
    def test_amoderate_content_success(self, mock_async_client_class):
        """Test successful async content moderation."""
        mock_async_client = Mock()
        mock_result = Mock()
        mock_result.flagged = False

        mock_response = Mock()
        mock_response.results = mock_result

        mock_async_client.moderations.create.return_value = mock_response
        mock_async_client_class.return_value = mock_async_client

        safety = LlamaStackSafety(base_url=self.base_url)

        import asyncio

        async def mock_amoderate():
            return await safety.amoderate_content("Clean async content")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(mock_amoderate())
        finally:
            loop.close()

        assert result.is_safe is True
        assert result.violations == []

        mock_async_client.moderations.create.assert_called_once_with(
            content="Clean async content", model=None
        )

    # @patch("langchain_llamastack.safety.AsyncLlamaStackClient")
    # def test_amoderate_content_violation(self, mock_async_client_class):
    #     """Test async content moderation with violation."""
    #     import asyncio
    #     from unittest.mock import AsyncMock

    #     mock_async_client = Mock()

    #     # Create a proper mock categories object that behaves like it has .items()
    #     mock_categories = Mock()
    #     mock_categories.items.return_value = [("violence", True), ("hate", False)]

    #     mock_result = Mock()
    #     mock_result.flagged = True  # Must be True for content to be flagged as unsafe
    #     mock_result.categories = mock_categories

    #     # Mock category_scores as well
    #     mock_category_scores = Mock()
    #     mock_category_scores.violence = 0.85
    #     mock_result.category_scores = mock_category_scores

    #     mock_response = Mock()
    #     mock_response.results = mock_result

    #     # Make the moderations.create method an async mock
    #     mock_async_client.moderations.create = AsyncMock(return_value=mock_response)
    #     mock_async_client_class.return_value = mock_async_client

    #     safety = LlamaStackSafety(base_url=self.base_url)

    #     async def mock_amoderate():
    #         return await safety.amoderate_content("Violent async content")

    #     loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(loop)
    #     try:
    #         result = loop.run_until_complete(mock_amoderate())
    #     finally:
    #         loop.close()

    #     assert result.is_safe is False
    #     assert len(result.violations) == 1
    #     assert result.violations[0]["category"] == "violence"

    @patch("langchain_llamastack.safety.AsyncLlamaStackClient")
    def test_amoderate_content_error(self, mock_async_client_class):
        """Test async content moderation with error."""
        mock_async_client = Mock()
        mock_async_client.moderations.create.side_effect = Exception(
            "Async moderation error"
        )
        mock_async_client_class.return_value = mock_async_client

        safety = LlamaStackSafety(base_url=self.base_url)

        import asyncio

        async def mock_amoderate():
            return await safety.amoderate_content("Test async content")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(mock_amoderate())
        finally:
            loop.close()

        assert result.is_safe is True  # Fails open
        assert "Async moderation failed: Async moderation error" in result.explanation

    @patch("langchain_llamastack.safety.LlamaStackClient")
    def test_client_initialization_lazy(self, mock_client_class):
        """Test that clients are initialized lazily."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        safety = LlamaStackSafety(base_url=self.base_url)

        # Client should not be initialized yet
        assert safety.client is None

        # Calling a method should initialize the client
        safety.check_content_safety("test")

        assert safety.client is not None

    @patch("langchain_llamastack.safety.AsyncLlamaStackClient")
    def test_async_client_initialization_lazy(self, mock_async_client_class):
        """Test that async clients are initialized lazily."""
        mock_async_client = Mock()
        mock_async_client_class.return_value = mock_async_client

        safety = LlamaStackSafety(base_url=self.base_url)

        # Async client should not be initialized yet
        assert safety.async_client is None

        # Calling an async method should initialize the client
        import asyncio

        async def mock_acheck():
            return await safety.acheck_content_safety("test")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(mock_acheck())
        finally:
            loop.close()

        assert safety.async_client is not None

    def test_content_safety_custom_shield_type(self):
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

    def test_content_moderation_custom_model(self):
        """Test content moderation with custom model."""
        with patch("langchain_llamastack.safety.LlamaStackClient") as mock_client_class:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.results = Mock()
            mock_response.results.flagged = False

            mock_client.moderations.create.return_value = mock_response
            mock_client_class.return_value = mock_client

            safety = LlamaStackSafety(
                base_url=self.base_url, moderation_model="custom_model"
            )
            safety.moderate_content("test content")

            mock_client.moderations.create.assert_called_with(
                content="test content", model="custom_model"
            )
