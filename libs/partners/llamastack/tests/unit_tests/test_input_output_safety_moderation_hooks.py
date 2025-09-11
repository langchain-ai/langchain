"""Unit tests for simplified input/output safety hooks."""

from typing import Any
from unittest.mock import Mock

# fmt: off
from langchain_llama_stack.input_output_safety_moderation_hooks import (
    create_safe_llm,
    SafeLLMWrapper,
)

# fmt: on
from langchain_llama_stack.safety import SafetyResult


class TestSafeLLMWrapper:
    """Test cases for SafeLLMWrapper."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = Mock()
        self.mock_safety_client = Mock()
        self.safe_llm = SafeLLMWrapper(self.mock_llm, self.mock_safety_client)

    def test_init(self) -> None:
        """Test SafeLLMWrapper initialization."""
        assert self.safe_llm.runnable == self.mock_llm
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

        assert "Execution failed: LLM failed" in result

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


class TestFactoryFunctions:
    """Test cases for factory functions."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = Mock()
        self.mock_safety_client = Mock()
        self.mock_safety_client.base_url = "http://localhost:8321"
        self.mock_safety_client.shield_type = "llama-guard"
        # Mock the list_shields method to return a list of shield identifiers
        self.mock_safety_client.list_shields.return_value = [
            "prompt-guard",
            "llama-guard",
            "content_safety",
        ]

    def test_create_safe_llm_input_only(self) -> None:
        """Test creating safe LLM with input checking only."""
        safe_llm = create_safe_llm(
            self.mock_llm, self.mock_safety_client, output_check=False
        )

        # Input hook may or may not be set depending on shield availability
        assert safe_llm.output_hook is None  # Should not be set

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

        # Mock LLM behavior
        self.mock_llm.invoke.return_value = "Test response"

        safe_llm = create_safe_llm(self.mock_llm, self.mock_safety_client)
        result = safe_llm.invoke("Test input")

        assert result == "Test response"
        # At least output hook should be called
        assert self.mock_safety_client.check_content_safety.call_count >= 1
