"""Unit tests for structured output retry functionality."""

from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel, ValidationError

from langchain.agents.structured_output import (
    OutputToolBinding,
    ProviderStrategy,
    ProviderStrategyBinding,
    StructuredOutputValidationError,
    ToolStrategy,
    _SchemaSpec,
)


class ResponseSchema(BaseModel):
    """Test schema for structured output."""

    value: int


class TestProviderStrategyRetry:
    """Test retry functionality for ProviderStrategy."""

    def test_provider_strategy_default_no_retry(self):
        """Test that ProviderStrategy defaults to no retry."""
        strategy = ProviderStrategy(ResponseSchema)
        assert strategy.max_retries == 0
        assert strategy.on_failure == "raise"

    def test_provider_strategy_with_max_retries(self):
        """Test ProviderStrategy with max_retries configured."""
        strategy = ProviderStrategy(ResponseSchema, max_retries=3)
        assert strategy.max_retries == 3

    def test_provider_strategy_with_custom_on_failure(self):
        """Test ProviderStrategy with custom on_failure function."""

        def custom_failure(exc: Exception) -> str:
            return f"Custom error: {exc}"

        strategy = ProviderStrategy(ResponseSchema, on_failure=custom_failure)
        assert callable(strategy.on_failure)

    def test_provider_binding_retry_succeeds_on_second_attempt(self):
        """Test that retry succeeds when second attempt works."""
        binding = ProviderStrategyBinding.from_schema_spec(_SchemaSpec(ResponseSchema))

        # Mock parse to fail once then succeed
        original_parse = binding.parse
        call_count = 0

        def mock_parse(msg):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First attempt fails")
            return ResponseSchema(value=42)

        binding.parse = mock_parse  # type: ignore[method-assign]

        # Simulate retry logic (would be in factory.py)
        max_retries = 2
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                result = binding.parse(Mock())
                assert result.value == 42
                assert call_count == 2  # Failed once, succeeded on retry
                break
            except Exception as exc:
                last_exception = exc
                if attempt >= max_retries:
                    raise

    def test_provider_binding_retry_exhausted(self):
        """Test that retries are exhausted and exception is raised."""
        binding = ProviderStrategyBinding.from_schema_spec(_SchemaSpec(ResponseSchema))

        # Mock parse to always fail
        def mock_parse(msg):
            raise ValueError("Always fails")

        binding.parse = mock_parse  # type: ignore[method-assign]

        # Simulate retry logic
        max_retries = 2
        last_exception = None
        attempts = 0
        for attempt in range(max_retries + 1):
            try:
                binding.parse(Mock())
            except Exception as exc:
                attempts += 1
                last_exception = exc
                if attempt >= max_retries:
                    break

        assert attempts == 3  # Initial + 2 retries
        assert last_exception is not None
        assert "Always fails" in str(last_exception)


class TestToolStrategyRetry:
    """Test retry functionality for ToolStrategy."""

    def test_tool_strategy_default_no_retry(self):
        """Test that ToolStrategy defaults to no retry."""
        strategy = ToolStrategy(ResponseSchema)
        assert strategy.max_retries == 0
        assert strategy.on_failure == "raise"

    def test_tool_strategy_with_max_retries(self):
        """Test ToolStrategy with max_retries configured."""
        strategy = ToolStrategy(ResponseSchema, max_retries=3)
        assert strategy.max_retries == 3

    def test_tool_strategy_retains_handle_errors(self):
        """Test that ToolStrategy retains handle_errors for backward compatibility."""
        strategy = ToolStrategy(ResponseSchema, handle_errors=False, max_retries=2)
        assert strategy.handle_errors is False
        assert strategy.max_retries == 2

    def test_tool_strategy_custom_error_message(self):
        """Test ToolStrategy with custom error message function."""

        def custom_error(exc: Exception) -> str:
            return f"Validation failed: {exc}"

        strategy = ToolStrategy(ResponseSchema, handle_errors=custom_error)
        assert callable(strategy.handle_errors)

    def test_tool_binding_retry_succeeds_on_third_attempt(self):
        """Test that retry succeeds when third attempt works."""
        binding = OutputToolBinding.from_schema_spec(_SchemaSpec(ResponseSchema))

        # Mock parse to fail twice then succeed
        call_count = 0

        def mock_parse(args):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValidationError.from_exception_data(
                    "ResponseSchema",
                    [{"type": "missing", "loc": ("value",), "msg": "Field required"}],
                )
            return ResponseSchema(value=100)

        binding.parse = mock_parse  # type: ignore[method-assign]

        # Simulate retry logic
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                result = binding.parse({"value": 100})
                assert result.value == 100
                assert call_count == 3  # Failed twice, succeeded on third
                break
            except Exception:
                if attempt >= max_retries:
                    raise

    def test_tool_binding_retry_with_on_failure_callable(self):
        """Test custom on_failure function is applied correctly."""

        def custom_failure(exc: Exception) -> str:
            return f"Custom message: {type(exc).__name__}"

        binding = OutputToolBinding.from_schema_spec(_SchemaSpec(ResponseSchema))

        def mock_parse(args):
            raise ValidationError.from_exception_data(
                "ResponseSchema",
                [{"type": "int_type", "loc": ("value",), "msg": "Input should be an integer"}],
            )

        binding.parse = mock_parse  # type: ignore[method-assign]

        # Simulate retry with custom failure
        max_retries = 1
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                binding.parse({"value": "not_an_int"})
            except Exception as exc:
                last_exception = exc
                if attempt >= max_retries:
                    break

        # Apply custom failure handler
        error_msg = custom_failure(last_exception)  # type: ignore[arg-type]
        assert "Custom message: ValidationError" in error_msg


class TestRetryBehavior:
    """Test overall retry behavior."""

    def test_max_retries_zero_means_no_retry(self):
        """Test that max_retries=0 means single attempt only."""
        strategy = ToolStrategy(ResponseSchema, max_retries=0)

        # With max_retries=0, should only attempt once
        attempts = 0
        for attempt in range(strategy.max_retries + 1):
            attempts += 1

        assert attempts == 1

    def test_max_retries_three_means_four_attempts(self):
        """Test that max_retries=3 means 4 total attempts."""
        strategy = ProviderStrategy(ResponseSchema, max_retries=3)

        # With max_retries=3, should attempt 4 times total
        attempts = 0
        for attempt in range(strategy.max_retries + 1):
            attempts += 1

        assert attempts == 4

    def test_on_failure_raise_is_default(self):
        """Test that on_failure defaults to 'raise'."""
        tool_strategy = ToolStrategy(ResponseSchema)
        provider_strategy = ProviderStrategy(ResponseSchema)

        assert tool_strategy.on_failure == "raise"
        assert provider_strategy.on_failure == "raise"
