"""Unit tests for simplified input/output safety hooks."""

from typing import Any
from unittest.mock import Mock

from langchain_llamastack.input_output_safety_moderation_hooks import (
    SafeLLMWrapper,
    create_input_only_safe_llm,
    create_output_only_safe_llm,
    create_safe_llm,
    create_safe_llm_with_all_hooks,
    create_safety_hook,
)
from langchain_llamastack.safety import SafetyResult


class TestSafeLLMWrapper:
    """Test cases for SafeLLMWrapper."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = Mock()
        self.mock_safety_client = Mock()
        self.safe_llm = SafeLLMWrapper(self.mock_llm, self.mock_safety_client)

    def test_init(self) -> None:
        """Test SafeLLMWrapper initialization."""
        assert self.safe_llm.llm == self.mock_llm
        assert self.safe_llm.safety_client == self.mock_safety_client
        assert self.safe_llm.input_hook is None
        assert self.safe_llm.output_hook is None

    def test_set_input_hook(self) -> None:
        """Test setting input hook."""
        mock_hook = Mock()
        self.safe_llm.set_input_hook(mock_hook)
        assert self.safe_llm.input_hook == mock_hook

    def test_set_output_hook(self) -> None:
        """Test setting output hook."""
        mock_hook = Mock()
        self.safe_llm.set_output_hook(mock_hook)
        assert self.safe_llm.output_hook == mock_hook

    def test_invoke_string_input(self) -> None:
        """Test invoke with string input and no hooks."""
        self.mock_llm.invoke.return_value = "Test response"

        result = self.safe_llm.invoke("Test input")

        assert result == "Test response"
        self.mock_llm.invoke.assert_called_once_with("Test input")

    def test_invoke_dict_input(self) -> None:
        """Test invoke with dict input."""
        self.mock_llm.invoke.return_value = "Test response"

        input_dict = {"input": "Test input"}
        result = self.safe_llm.invoke(input_dict)

        assert result == "Test response"
        self.mock_llm.invoke.assert_called_once_with("Test input")

    def test_invoke_dict_input_question_key(self) -> None:
        """Test invoke with dict input using 'question' key."""
        self.mock_llm.invoke.return_value = "Test response"

        input_dict = {"question": "Test question"}
        result = self.safe_llm.invoke(input_dict)

        assert result == "Test response"
        self.mock_llm.invoke.assert_called_once_with("Test question")

    def test_invoke_input_blocked(self) -> None:
        """Test invoke blocked by input hook."""
        # Setup input hook that blocks
        mock_hook = Mock()
        mock_hook.return_value = SafetyResult(
            is_safe=False, violations=[{"reason": "Inappropriate content"}]
        )
        self.safe_llm.set_input_hook(mock_hook)

        result = self.safe_llm.invoke("Bad input")

        assert "Input blocked by safety system" in result
        assert "Inappropriate content" in result
        mock_hook.assert_called_once_with("Bad input")
        self.mock_llm.invoke.assert_not_called()

    def test_invoke_output_blocked(self) -> None:
        """Test invoke blocked by output hook."""
        # Setup LLM response
        self.mock_llm.invoke.return_value = "Unsafe response"

        # Setup output hook that blocks
        mock_hook = Mock()
        mock_hook.return_value = SafetyResult(
            is_safe=False, violations=[{"reason": "Unsafe output"}]
        )
        self.safe_llm.set_output_hook(mock_hook)

        result = self.safe_llm.invoke("Test input")

        assert "Output blocked by safety system" in result
        assert "Unsafe output" in result
        mock_hook.assert_called_once_with("Unsafe response")

    def test_invoke_llm_error(self) -> None:
        """Test invoke with LLM error."""
        self.mock_llm.invoke.side_effect = Exception("LLM failed")

        result = self.safe_llm.invoke("Test input")

        assert "LLM execution failed: LLM failed" in result

    def test_invoke_all_hooks_pass(self) -> None:
        """Test invoke with all hooks passing."""
        # Setup LLM
        self.mock_llm.invoke.return_value = "Clean response"

        # Setup hooks to pass
        mock_input_hook = Mock(return_value=SafetyResult(is_safe=True))
        mock_output_hook = Mock(return_value=SafetyResult(is_safe=True))

        self.safe_llm.set_input_hook(mock_input_hook)
        self.safe_llm.set_output_hook(mock_output_hook)

        result = self.safe_llm.invoke("Clean input")

        assert result == "Clean response"
        mock_input_hook.assert_called_once_with("Clean input")
        mock_output_hook.assert_called_once_with("Clean response")

    def test_invoke_llm_response_with_content_attr(self) -> None:
        """Test invoke with LLM response having content attribute."""
        # Mock LLM response with content attribute
        mock_response = Mock()
        mock_response.content = "Response content"
        self.mock_llm.invoke.return_value = mock_response

        result = self.safe_llm.invoke("Test input")

        assert result == "Response content"

    def test_ainvoke_input_blocked(self) -> None:
        """Test async invoke blocked by input hook."""
        mock_hook = Mock()
        mock_hook.return_value = SafetyResult(
            is_safe=False, violations=[{"reason": "Async safety block"}]
        )
        self.safe_llm.set_input_hook(mock_hook)

        import asyncio

        async def mock_ainvoke() -> Any:
            return await self.safe_llm.ainvoke("Bad async input")

        result = asyncio.run(mock_ainvoke())

        assert "Input blocked by safety system" in result
        assert "Async safety block" in result
        self.mock_llm.ainvoke.assert_not_called()


class TestHookCreators:
    """Test cases for hook creator functions."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_safety_client = Mock()

    def test_create_safety_hook_input_success(self) -> None:
        """Test creating input safety hook with successful check."""
        self.mock_safety_client.check_content_safety.return_value = SafetyResult(
            is_safe=True, confidence_score=0.95
        )

        hook = create_safety_hook(self.mock_safety_client, "input")
        result = hook("Clean input")

        assert result.is_safe is True
        assert result.confidence_score == 0.95
        self.mock_safety_client.check_content_safety.assert_called_once_with(
            "Clean input"
        )

    def test_create_safety_hook_input_violation(self) -> None:
        """Test creating input safety hook with violation."""
        self.mock_safety_client.check_content_safety.return_value = SafetyResult(
            is_safe=False, violations=[{"category": "hate"}]
        )

        hook = create_safety_hook(self.mock_safety_client, "input")
        result = hook("Bad input")

        assert result.is_safe is False
        assert len(result.violations) == 1

    def test_create_safety_hook_input_error_fail_open(self) -> None:
        """Test creating input safety hook with error (fails open)."""
        self.mock_safety_client.check_content_safety.side_effect = Exception(
            "Safety check failed"
        )

        hook = create_safety_hook(self.mock_safety_client, "input")
        result = hook("Test input")

        assert result.is_safe is True  # Fail open for input
        assert "Safety check failed" in result.explanation

    def test_create_safety_hook_output_success(self) -> None:
        """Test creating output safety hook with successful check."""
        self.mock_safety_client.check_content_safety.return_value = SafetyResult(
            is_safe=True
        )

        hook = create_safety_hook(self.mock_safety_client, "output")
        result = hook("Safe output")

        assert result.is_safe is True
        self.mock_safety_client.check_content_safety.assert_called_once_with(
            "Safe output"
        )

    def test_create_safety_hook_output_error_fail_closed(self) -> None:
        """Test creating output safety hook with error (fails closed)."""
        self.mock_safety_client.check_content_safety.side_effect = Exception(
            "Output safety failed"
        )

        hook = create_safety_hook(self.mock_safety_client, "output")
        result = hook("Test output")

        assert result.is_safe is False  # Fail closed for output
        assert "Safety check failed" in result.explanation


class TestFactoryFunctions:
    """Test cases for factory functions."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = Mock()
        self.mock_safety_client = Mock()

    def test_create_safe_llm_default(self) -> None:
        """Test creating safe LLM with default settings (both hooks)."""
        safe_llm = create_safe_llm(self.mock_llm, self.mock_safety_client)

        assert isinstance(safe_llm, SafeLLMWrapper)
        assert safe_llm.llm == self.mock_llm
        assert safe_llm.safety_client == self.mock_safety_client
        assert safe_llm.input_hook is not None  # Should be set
        assert safe_llm.output_hook is not None  # Should be set

    def test_create_safe_llm_input_only(self) -> None:
        """Test creating safe LLM with input checking only."""
        safe_llm = create_safe_llm(
            self.mock_llm, self.mock_safety_client, output_check=False
        )

        assert safe_llm.input_hook is not None  # Should be set
        assert safe_llm.output_hook is None  # Should not be set

    def test_create_safe_llm_output_only(self) -> None:
        """Test creating safe LLM with output checking only."""
        safe_llm = create_safe_llm(
            self.mock_llm, self.mock_safety_client, input_check=False
        )

        assert safe_llm.input_hook is None  # Should not be set
        assert safe_llm.output_hook is not None  # Should be set

    def test_create_safe_llm_no_hooks(self) -> None:
        """Test creating safe LLM with no hooks."""
        safe_llm = create_safe_llm(
            self.mock_llm,
            self.mock_safety_client,
            input_check=False,
            output_check=False,
        )

        assert safe_llm.input_hook is None  # Should not be set
        assert safe_llm.output_hook is None  # Should not be set

    def test_create_safe_llm_with_all_hooks(self) -> None:
        """Test create_safe_llm_with_all_hooks factory."""
        safe_llm = create_safe_llm_with_all_hooks(
            self.mock_llm, self.mock_safety_client
        )

        assert isinstance(safe_llm, SafeLLMWrapper)
        assert safe_llm.input_hook is not None
        assert safe_llm.output_hook is not None

    def test_create_input_only_safe_llm(self) -> None:
        """Test create_input_only_safe_llm factory."""
        safe_llm = create_input_only_safe_llm(self.mock_llm, self.mock_safety_client)

        assert safe_llm.input_hook is not None
        assert safe_llm.output_hook is None

    def test_create_output_only_safe_llm(self) -> None:
        """Test create_output_only_safe_llm factory."""
        safe_llm = create_output_only_safe_llm(self.mock_llm, self.mock_safety_client)

        assert safe_llm.input_hook is None
        assert safe_llm.output_hook is not None

    def test_factory_functions_behavior(self) -> None:
        """Test that factory function hooks work correctly."""
        # Mock safety client behavior
        self.mock_safety_client.check_content_safety.return_value = SafetyResult(
            is_safe=True
        )

        # Mock LLM behavior
        self.mock_llm.invoke.return_value = "Test response"

        safe_llm = create_safe_llm(self.mock_llm, self.mock_safety_client)
        result = safe_llm.invoke("Test input")

        assert result == "Test response"
        # Should be called twice: once for input, once for output
        assert self.mock_safety_client.check_content_safety.call_count == 2

    def test_hook_execution_order(self) -> None:
        """Test that hooks are executed in the correct order."""
        # Track call order
        call_order = []

        def track_input_call(content: Any) -> Any:
            call_order.append(f"input_{content}")
            return SafetyResult(is_safe=True)

        def track_output_call(content: Any) -> Any:
            call_order.append(f"output_{content}")
            return SafetyResult(is_safe=True)

        self.mock_safety_client.check_content_safety.side_effect = [
            SafetyResult(is_safe=True),  # Input call
            SafetyResult(is_safe=True),  # Output call
        ]

        # Create hooks with tracking
        safe_llm = create_safe_llm(self.mock_llm, self.mock_safety_client)
        # Replace hooks with tracking versions
        safe_llm.set_input_hook(track_input_call)
        safe_llm.set_output_hook(track_output_call)

        self.mock_llm.invoke.return_value = "Test response"
        safe_llm.invoke("Test input")

        # Verify execution order: input -> LLM -> output
        expected_order = [
            "input_Test input",
            "output_Test response",
        ]
        assert call_order == expected_order
