"""Unit tests for input/output safety moderation hooks."""

from unittest.mock import Mock

from langchain_llamastack.input_output_safety_moderation_hooks import (
    SafeLLMWrapper,
    create_input_moderation_hook,
    create_input_only_safe_llm,
    create_input_safety_hook,
    create_moderation_only_llm,
    create_output_moderation_hook,
    create_output_only_safe_llm,
    create_output_safety_hook,
    create_safe_llm_with_all_hooks,
    create_safety_only_llm,
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
        assert self.safe_llm.input_safety_hook is None
        assert self.safe_llm.input_moderation_hook is None
        assert self.safe_llm.output_safety_hook is None
        assert self.safe_llm.output_moderation_hook is None

    def test_set_input_safety_hook(self) -> None:
        """Test setting input safety hook."""
        mock_hook = Mock()
        self.safe_llm.set_input_safety_hook(mock_hook)
        assert self.safe_llm.input_safety_hook == mock_hook

    def test_set_input_moderation_hook(self) -> None:
        """Test setting input moderation hook."""
        mock_hook = Mock()
        self.safe_llm.set_input_moderation_hook(mock_hook)
        assert self.safe_llm.input_moderation_hook == mock_hook

    def test_set_output_safety_hook(self) -> None:
        """Test setting output safety hook."""
        mock_hook = Mock()
        self.safe_llm.set_output_safety_hook(mock_hook)
        assert self.safe_llm.output_safety_hook == mock_hook

    def test_set_output_moderation_hook(self) -> None:
        """Test setting output moderation hook."""
        mock_hook = Mock()
        self.safe_llm.set_output_moderation_hook(mock_hook)
        assert self.safe_llm.output_moderation_hook == mock_hook

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

    def test_invoke_dict_input_fallback(self) -> None:
        """Test invoke with dict input fallback to string conversion."""
        self.mock_llm.invoke.return_value = "Test response"

        input_dict = {"other": "value"}
        result = self.safe_llm.invoke(input_dict)

        assert result == "Test response"
        self.mock_llm.invoke.assert_called_once_with("{'other': 'value'}")

    def test_invoke_input_safety_blocked(self) -> None:
        """Test invoke blocked by input safety hook."""
        # Setup input safety hook that blocks
        mock_hook = Mock()
        mock_hook.return_value = SafetyResult(
            is_safe=False, violations=[{"reason": "Inappropriate content"}]
        )
        self.safe_llm.set_input_safety_hook(mock_hook)

        result = self.safe_llm.invoke("Bad input")

        assert "Input blocked by safety system" in result
        assert "Inappropriate content" in result
        mock_hook.assert_called_once_with("Bad input")
        self.mock_llm.invoke.assert_not_called()

    def test_invoke_input_moderation_blocked(self) -> None:
        """Test invoke blocked by input moderation hook."""
        # Setup hooks
        mock_safety_hook = Mock()
        mock_safety_hook.return_value = SafetyResult(is_safe=True)
        mock_moderation_hook = Mock()
        mock_moderation_hook.return_value = SafetyResult(
            is_safe=False, violations=[{"reason": "Policy violation"}]
        )

        self.safe_llm.set_input_safety_hook(mock_safety_hook)
        self.safe_llm.set_input_moderation_hook(mock_moderation_hook)

        result = self.safe_llm.invoke("Bad input")

        assert "Input blocked by moderation system" in result
        assert "Policy violation" in result
        self.mock_llm.invoke.assert_not_called()

    def test_invoke_llm_error(self) -> None:
        """Test invoke with LLM error."""
        # Setup hooks to pass
        self.mock_llm.invoke.side_effect = Exception("LLM failed")

        result = self.safe_llm.invoke("Test input")

        assert "LLM execution failed: LLM failed" in result

    def test_invoke_output_safety_blocked(self) -> None:
        """Test invoke blocked by output safety hook."""
        # Setup LLM response
        self.mock_llm.invoke.return_value = "Unsafe response"

        # Setup output safety hook that blocks
        mock_hook = Mock()
        mock_hook.return_value = SafetyResult(
            is_safe=False, violations=[{"reason": "Unsafe output"}]
        )
        self.safe_llm.set_output_safety_hook(mock_hook)

        result = self.safe_llm.invoke("Test input")

        assert "Output blocked by safety system" in result
        assert "Unsafe output" in result
        mock_hook.assert_called_once_with("Unsafe response")

    def test_invoke_output_moderation_blocked(self) -> None:
        """Test invoke blocked by output moderation hook."""
        # Setup LLM response
        self.mock_llm.invoke.return_value = "Policy violation response"

        # Setup output moderation hook that blocks
        mock_hook = Mock()
        mock_hook.return_value = SafetyResult(
            is_safe=False, violations=[{"reason": "Content policy violation"}]
        )
        self.safe_llm.set_output_moderation_hook(mock_hook)

        result = self.safe_llm.invoke("Test input")

        assert "Output blocked by moderation system" in result
        assert "Content policy violation" in result
        mock_hook.assert_called_once_with("Policy violation response")

    def test_invoke_llm_response_with_content_attr(self) -> None:
        """Test invoke with LLM response having content attribute."""
        # Mock LLM response with content attribute
        mock_response = Mock()
        mock_response.content = "Response content"
        self.mock_llm.invoke.return_value = mock_response

        result = self.safe_llm.invoke("Test input")

        assert result == "Response content"

    # def test_invoke_llm_response_object_conversion(self):
    #     """Test invoke with LLM response object conversion."""
    #     # Mock LLM response object without content attribute
    #     mock_response = Mock()
    #     mock_response.__str__ = Mock(return_value="Converted response")
    #     self.mock_llm.invoke.return_value = mock_response

    #     result = self.safe_llm.invoke("Test input")

    #     assert result == "Converted response"

    def test_invoke_all_hooks_pass(self) -> None:
        """Test invoke with all hooks passing."""
        # Setup LLM
        self.mock_llm.invoke.return_value = "Clean response"

        # Setup all hooks to pass
        mock_input_safety = Mock(return_value=SafetyResult(is_safe=True))
        mock_input_moderation = Mock(return_value=SafetyResult(is_safe=True))
        mock_output_safety = Mock(return_value=SafetyResult(is_safe=True))
        mock_output_moderation = Mock(return_value=SafetyResult(is_safe=True))

        self.safe_llm.set_input_safety_hook(mock_input_safety)
        self.safe_llm.set_input_moderation_hook(mock_input_moderation)
        self.safe_llm.set_output_safety_hook(mock_output_safety)
        self.safe_llm.set_output_moderation_hook(mock_output_moderation)

        result = self.safe_llm.invoke("Clean input")

        assert result == "Clean response"
        mock_input_safety.assert_called_once_with("Clean input")
        mock_input_moderation.assert_called_once_with("Clean input")
        mock_output_safety.assert_called_once_with("Clean response")
        mock_output_moderation.assert_called_once_with("Clean response")

    # def test_ainvoke_string_input(self):
    #     """Test async invoke with string input."""
    #     self.mock_llm.ainvoke.return_value = "Async response"

    #     import asyncio

    #     async def mock_ainvoke():
    #         return await self.safe_llm.ainvoke("Test input")

    #     loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(loop)
    #     try:
    #         result = loop.run_until_complete(mock_ainvoke())
    #     finally:
    #         loop.close()

    #     assert result == "Async response"
    #     self.mock_llm.ainvoke.assert_called_once_with("Test input")

    def test_ainvoke_input_safety_blocked(self) -> None:
        """Test async invoke blocked by input safety hook."""
        mock_hook = Mock()
        mock_hook.return_value = SafetyResult(
            is_safe=False, violations=[{"reason": "Async safety block"}]
        )
        self.safe_llm.set_input_safety_hook(mock_hook)

        import asyncio

        async def mock_ainvoke():
            return await self.safe_llm.ainvoke("Bad async input")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(mock_ainvoke())
        finally:
            loop.close()

        assert "Input blocked by safety system" in result
        assert "Async safety block" in result
        self.mock_llm.ainvoke.assert_not_called()

    def test_ainvoke_llm_error(self) -> None:
        """Test async invoke with LLM error."""
        self.mock_llm.ainvoke.side_effect = Exception("Async LLM failed")

        import asyncio

        async def mock_ainvoke():
            return await self.safe_llm.ainvoke("Test input")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(mock_ainvoke())
        finally:
            loop.close()

        assert "LLM execution failed: Async LLM failed" in result

    # def test_ainvoke_output_safety_blocked(self):
    #     """Test async invoke blocked by output safety hook."""
    #     self.mock_llm.ainvoke.return_value = "Unsafe async response"

    #     mock_hook = Mock()
    #     mock_hook.return_value = SafetyResult(
    #         is_safe=False, violations=[{"reason": "Async unsafe output"}]
    #     )
    #     self.safe_llm.set_output_safety_hook(mock_hook)

    #     import asyncio

    #     async def mock_ainvoke():
    #         return await self.safe_llm.ainvoke("Test input")

    #     loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(loop)
    #     try:
    #         result = loop.run_until_complete(mock_ainvoke())
    #     finally:
    #         loop.close()

    #     assert "Output blocked by safety system" in result
    #     assert "Async unsafe output" in result


class TestHookCreators:
    """Test cases for hook creator functions."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_safety_client = Mock()

    def test_create_input_safety_hook_success(self) -> None:
        """Test creating input safety hook with successful check."""
        self.mock_safety_client.check_content_safety.return_value = SafetyResult(
            is_safe=True, confidence_score=0.95
        )

        hook = create_input_safety_hook(self.mock_safety_client)
        result = hook("Clean input")

        assert result.is_safe is True
        assert result.confidence_score == 0.95
        self.mock_safety_client.check_content_safety.assert_called_once_with(
            "Clean input"
        )

    def test_create_input_safety_hook_violation(self) -> None:
        """Test creating input safety hook with violation."""
        self.mock_safety_client.check_content_safety.return_value = SafetyResult(
            is_safe=False, violations=[{"category": "hate"}]
        )

        hook = create_input_safety_hook(self.mock_safety_client)
        result = hook("Bad input")

        assert result.is_safe is False
        assert len(result.violations) == 1

    def test_create_input_safety_hook_error(self) -> None:
        """Test creating input safety hook with error (fail open)."""
        self.mock_safety_client.check_content_safety.side_effect = Exception(
            "Safety check failed"
        )

        hook = create_input_safety_hook(self.mock_safety_client)
        result = hook("Test input")

        assert result.is_safe is True  # Fail open for input safety
        assert "Input safety check failed" in result.explanation

    def test_create_input_moderation_hook_success(self) -> None:
        """Test creating input moderation hook with successful check."""
        self.mock_safety_client.moderate_content.return_value = SafetyResult(
            is_safe=True, violations=[]
        )

        hook = create_input_moderation_hook(self.mock_safety_client)
        result = hook("Clean input")

        assert result.is_safe is True
        self.mock_safety_client.moderate_content.assert_called_once_with("Clean input")

    def test_create_input_moderation_hook_error(self) -> None:
        """Test creating input moderation hook with error (fail open)."""
        self.mock_safety_client.moderate_content.side_effect = Exception(
            "Moderation failed"
        )

        hook = create_input_moderation_hook(self.mock_safety_client)
        result = hook("Test input")

        assert result.is_safe is True  # Fail open for input moderation
        assert "Input moderation failed" in result.explanation

    def test_create_output_safety_hook_success(self) -> None:
        """Test creating output safety hook with successful check."""
        self.mock_safety_client.check_content_safety.return_value = SafetyResult(
            is_safe=True
        )

        hook = create_output_safety_hook(self.mock_safety_client)
        result = hook("Safe output")

        assert result.is_safe is True
        self.mock_safety_client.check_content_safety.assert_called_once_with(
            "Safe output"
        )

    def test_create_output_safety_hook_error(self) -> None:
        """Test creating output safety hook with error (fail closed)."""
        self.mock_safety_client.check_content_safety.side_effect = Exception(
            "Output safety failed"
        )

        hook = create_output_safety_hook(self.mock_safety_client)
        result = hook("Test output")

        assert result.is_safe is False  # Fail closed for output safety
        assert "Output safety check failed" in result.explanation

    def test_create_output_moderation_hook_success(self) -> None:
        """Test creating output moderation hook with successful check."""
        self.mock_safety_client.moderate_content.return_value = SafetyResult(
            is_safe=True
        )

        hook = create_output_moderation_hook(self.mock_safety_client)
        result = hook("Clean output")

        assert result.is_safe is True

    def test_create_output_moderation_hook_error(self) -> None:
        """Test creating output moderation hook with error (fail closed)."""
        self.mock_safety_client.moderate_content.side_effect = Exception(
            "Output moderation failed"
        )

        hook = create_output_moderation_hook(self.mock_safety_client)
        result = hook("Test output")

        assert result.is_safe is False  # Fail closed for output moderation
        assert "Output moderation failed" in result.explanation


class TestFactoryFunctions:
    """Test cases for factory functions."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = Mock()
        self.mock_safety_client = Mock()

    def test_create_safe_llm_with_all_hooks(self) -> None:
        """Test creating safe LLM with all hooks."""
        safe_llm = create_safe_llm_with_all_hooks(
            self.mock_llm, self.mock_safety_client
        )

        assert isinstance(safe_llm, SafeLLMWrapper)
        assert safe_llm.llm == self.mock_llm
        assert safe_llm.safety_client == self.mock_safety_client
        assert safe_llm.input_safety_hook is not None
        assert safe_llm.input_moderation_hook is not None
        assert safe_llm.output_safety_hook is not None
        assert safe_llm.output_moderation_hook is not None

    def test_create_input_only_safe_llm(self) -> None:
        """Test creating safe LLM with input hooks only."""
        safe_llm = create_input_only_safe_llm(self.mock_llm, self.mock_safety_client)

        assert isinstance(safe_llm, SafeLLMWrapper)
        assert safe_llm.input_safety_hook is not None
        assert safe_llm.input_moderation_hook is not None
        assert safe_llm.output_safety_hook is None
        assert safe_llm.output_moderation_hook is None

    def test_create_output_only_safe_llm(self) -> None:
        """Test creating safe LLM with output hooks only."""
        safe_llm = create_output_only_safe_llm(self.mock_llm, self.mock_safety_client)

        assert isinstance(safe_llm, SafeLLMWrapper)
        assert safe_llm.input_safety_hook is None
        assert safe_llm.input_moderation_hook is None
        assert safe_llm.output_safety_hook is not None
        assert safe_llm.output_moderation_hook is not None

    def test_create_safety_only_llm(self) -> None:
        """Test creating safe LLM with safety hooks only."""
        safe_llm = create_safety_only_llm(self.mock_llm, self.mock_safety_client)

        assert isinstance(safe_llm, SafeLLMWrapper)
        assert safe_llm.input_safety_hook is not None
        assert safe_llm.input_moderation_hook is None
        assert safe_llm.output_safety_hook is not None
        assert safe_llm.output_moderation_hook is None

    def test_create_moderation_only_llm(self) -> None:
        """Test creating safe LLM with moderation hooks only."""
        safe_llm = create_moderation_only_llm(self.mock_llm, self.mock_safety_client)

        assert isinstance(safe_llm, SafeLLMWrapper)
        assert safe_llm.input_safety_hook is None
        assert safe_llm.input_moderation_hook is not None
        assert safe_llm.output_safety_hook is None
        assert safe_llm.output_moderation_hook is not None

    def test_factory_functions_hook_behavior(self) -> None:
        """Test that factory function hooks work correctly."""
        # Mock safety client behavior
        self.mock_safety_client.check_content_safety.return_value = SafetyResult(
            is_safe=True
        )
        self.mock_safety_client.moderate_content.return_value = SafetyResult(
            is_safe=True
        )

        # Mock LLM behavior
        self.mock_llm.invoke.return_value = "Test response"

        safe_llm = create_safe_llm_with_all_hooks(
            self.mock_llm, self.mock_safety_client
        )
        result = safe_llm.invoke("Test input")

        assert result == "Test response"
        # Verify all hooks were called
        # Input + output safety
        assert self.mock_safety_client.check_content_safety.call_count == 2
        # Input + output moderation
        assert self.mock_safety_client.moderate_content.call_count == 2

    def test_hook_execution_order(self) -> None:
        """Test that hooks are executed in the correct order."""
        # Track call order
        call_order = []

        def track_safety_call(content):
            call_order.append(f"safety_{content}")
            return SafetyResult(is_safe=True)

        def track_moderation_call(content):
            call_order.append(f"moderation_{content}")
            return SafetyResult(is_safe=True)

        self.mock_safety_client.check_content_safety.side_effect = track_safety_call
        self.mock_safety_client.moderate_content.side_effect = track_moderation_call
        self.mock_llm.invoke.return_value = "Test response"

        safe_llm = create_safe_llm_with_all_hooks(
            self.mock_llm, self.mock_safety_client
        )
        safe_llm.invoke("Test input")

        # Verify execution order: input safety -> input moderation
        # -> LLM -> output safety -> output moderation
        expected_order = [
            "safety_Test input",  # Input safety
            "moderation_Test input",  # Input moderation
            "safety_Test response",  # Output safety
            "moderation_Test response",  # Output moderation
        ]
        assert call_order == expected_order
