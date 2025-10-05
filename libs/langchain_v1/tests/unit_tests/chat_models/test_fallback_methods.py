"""Test fallback methods in BaseChatModel."""

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from typing_extensions import override


class FakeChatModelForFallbackTesting(BaseChatModel):
    """A fake chat model for testing fallback methods."""

    responses: list[str]
    error_on_invoke: bool = False

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat result."""
        if self.error_on_invoke:
            msg = "Simulated error for testing"
            raise ValueError(msg)

        response_text = self.responses[0] if self.responses else "default response"
        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "fake-fallback-test-model"


class TestBaseChatModelFallbackMethods:
    """Test suite for BaseChatModel fallback methods."""

    def test_initial_fallback_state(self) -> None:
        """Test that a new chat model has no fallback initially."""
        model = FakeChatModelForFallbackTesting(responses=["test"])
        assert model.get_fallback() is None

    def test_set_fallback_with_runnable_with_fallbacks(self) -> None:
        """Test setting fallback with a RunnableWithFallbacks object."""
        primary_model = FakeChatModelForFallbackTesting(responses=["primary"])
        fallback_model = FakeChatModelForFallbackTesting(responses=["fallback"])

        # Create a RunnableWithFallbacks object
        fallback_runnable = primary_model.with_fallbacks([fallback_model])

        # Set the fallback
        test_model = FakeChatModelForFallbackTesting(responses=["test"])
        result = test_model.set_fallback(fallback_runnable)

        # Should return self for chaining
        assert result is test_model

        # Should store the fallback
        assert test_model.get_fallback() is fallback_runnable

    def test_set_fallback_with_none(self) -> None:
        """Test setting fallback to None."""
        model = FakeChatModelForFallbackTesting(responses=["test"])

        # First set a fallback
        fallback_model = FakeChatModelForFallbackTesting(responses=["fallback"])
        fallback_runnable = model.with_fallbacks([fallback_model])
        model.set_fallback(fallback_runnable)

        # Verify it's set
        assert model.get_fallback() is not None

        # Now set to None
        result = model.set_fallback(None)

        # Should return self for chaining
        assert result is model

        # Should clear the fallback
        assert model.get_fallback() is None

    def test_get_fallback_returns_correct_object(self) -> None:
        """Test that get_fallback returns the correct fallback object."""
        model = FakeChatModelForFallbackTesting(responses=["test"])

        # Initially None
        assert model.get_fallback() is None

        # Set a fallback and verify
        fallback_model = FakeChatModelForFallbackTesting(responses=["fallback"])
        fallback_runnable = model.with_fallbacks([fallback_model])
        model.set_fallback(fallback_runnable)

        retrieved_fallback = model.get_fallback()
        assert retrieved_fallback is fallback_runnable

    def test_reset_fallback(self) -> None:
        """Test that _reset_fallback clears the fallback and returns self."""
        model = FakeChatModelForFallbackTesting(responses=["test"])

        # Set a fallback first
        fallback_model = FakeChatModelForFallbackTesting(responses=["fallback"])
        fallback_runnable = model.with_fallbacks([fallback_model])
        model.set_fallback(fallback_runnable)

        # Verify it's set
        assert model.get_fallback() is not None

        # Reset the fallback
        result = model._reset_fallback()

        # Should return self
        assert result is model

        # Should clear the fallback
        assert model.get_fallback() is None

    def test_fallback_integration_with_invoke(self) -> None:
        """Test that fallback is used correctly in invoke method."""
        # Create a working fallback model
        fallback_model = FakeChatModelForFallbackTesting(responses=["fallback response"])

        # Create a primary model that will error
        primary_model = FakeChatModelForFallbackTesting(responses=["primary"], error_on_invoke=True)

        # Create fallback runnable using the working fallback model
        fallback_runnable = fallback_model.with_fallbacks([])

        # Set the fallback on the primary model
        primary_model.set_fallback(fallback_runnable)

        # Call invoke - should use fallback since primary model errors
        result = primary_model.invoke("test message")

        # Verify fallback was used (should get fallback response)
        assert result.content == "fallback response"

        # Verify fallback was reset after use
        assert primary_model.get_fallback() is None

    async def test_fallback_integration_with_ainvoke(self) -> None:
        """Test that fallback is used correctly in ainvoke method."""
        # Create a working fallback model
        fallback_model = FakeChatModelForFallbackTesting(responses=["async fallback response"])

        # Create a primary model that will error
        primary_model = FakeChatModelForFallbackTesting(responses=["primary"], error_on_invoke=True)

        # Create fallback runnable using the working fallback model
        fallback_runnable = fallback_model.with_fallbacks([])

        # Set the fallback on the primary model
        primary_model.set_fallback(fallback_runnable)

        # Call ainvoke - should use fallback since primary model errors
        result = await primary_model.ainvoke("test message")

        # Verify fallback was used (should get fallback response)
        assert result.content == "async fallback response"

        # Verify fallback was reset after use
        assert primary_model.get_fallback() is None

    def test_fallback_integration_with_stream(self) -> None:
        """Test that fallback is used correctly in stream method."""
        # Create a working fallback model
        fallback_model = FakeChatModelForFallbackTesting(responses=["stream fallback response"])

        # Create a primary model that will error
        primary_model = FakeChatModelForFallbackTesting(responses=["primary"], error_on_invoke=True)

        # Create fallback runnable using the working fallback model
        fallback_runnable = fallback_model.with_fallbacks([])

        # Set the fallback on the primary model
        primary_model.set_fallback(fallback_runnable)

        # Call stream - should use fallback since primary model errors
        chunks = list(primary_model.stream("test message"))

        # Verify fallback was used (should get fallback response as a single chunk)
        assert len(chunks) == 1
        assert chunks[0].content == "stream fallback response"

        # Verify fallback was reset after use
        assert primary_model.get_fallback() is None

    async def test_fallback_integration_with_astream(self) -> None:
        """Test that fallback is used correctly in astream method."""
        # Create a working fallback model
        fallback_model = FakeChatModelForFallbackTesting(
            responses=["async stream fallback response"]
        )

        # Create a primary model that will error
        primary_model = FakeChatModelForFallbackTesting(responses=["primary"], error_on_invoke=True)

        # Create fallback runnable using the working fallback model
        fallback_runnable = fallback_model.with_fallbacks([])

        # Set the fallback on the primary model
        primary_model.set_fallback(fallback_runnable)

        # Call astream - should use fallback since primary model errors
        chunks = [chunk async for chunk in primary_model.astream("test message")]

        # Verify fallback was used (should get fallback response as a single chunk)
        assert len(chunks) == 1
        assert chunks[0].content == "async stream fallback response"

        # Verify fallback was reset after use
        assert primary_model.get_fallback() is None

    def test_no_fallback_normal_operation(self) -> None:
        """Test that methods work normally when no fallback is set."""
        model = FakeChatModelForFallbackTesting(responses=["normal response"])

        # Verify no fallback is set
        assert model.get_fallback() is None

        # Normal invoke should work
        result = model.invoke("test message")
        assert result.content == "normal response"

        # Fallback should still be None
        assert model.get_fallback() is None

    def test_fallback_used_when_set(self) -> None:
        """Test that fallback is used when set, regardless of primary model success."""
        # Create a working primary model
        primary_model = FakeChatModelForFallbackTesting(responses=["primary success"])

        # Create a fallback model
        fallback_model = FakeChatModelForFallbackTesting(responses=["fallback response"])

        # Create and set fallback
        fallback_runnable = fallback_model.with_fallbacks([])
        primary_model.set_fallback(fallback_runnable)

        # Call invoke - should use fallback since it's set
        result = primary_model.invoke("test message")

        assert result.content == "fallback response"

        # Verify fallback was reset after use
        assert primary_model.get_fallback() is None

    def test_method_chaining(self) -> None:
        """Test that set_fallback and _reset_fallback support method chaining."""
        model = FakeChatModelForFallbackTesting(responses=["test"])
        fallback_model = FakeChatModelForFallbackTesting(responses=["fallback"])
        fallback_runnable = fallback_model.with_fallbacks([])

        # Test chaining set_fallback
        result = model.set_fallback(fallback_runnable).set_fallback(None)
        assert result is model
        assert model.get_fallback() is None

        # Test chaining _reset_fallback
        model.set_fallback(fallback_runnable)
        result = model._reset_fallback()
        assert result is model
        assert model.get_fallback() is None

    def test_fallback_field_excluded_from_serialization(self) -> None:
        """Test that fallback field is excluded from model serialization."""
        model = FakeChatModelForFallbackTesting(responses=["test"])
        fallback_model = FakeChatModelForFallbackTesting(responses=["fallback"])
        fallback_runnable = fallback_model.with_fallbacks([])

        # Set fallback
        model.set_fallback(fallback_runnable)

        # Check that fallback is not in the model dict (excluded=True)
        model_dict = model.model_dump()
        assert "fallback" not in model_dict

        # But the fallback should still be accessible via get_fallback()
        assert model.get_fallback() is fallback_runnable

    def test_fallback_with_different_input_types(self) -> None:
        """Test fallback works with different input types (str, list, etc.)."""
        # Create a working fallback model
        fallback_model = FakeChatModelForFallbackTesting(responses=["fallback used"])

        # Create a primary model that will error
        primary_model = FakeChatModelForFallbackTesting(responses=["primary"], error_on_invoke=True)

        # Create and set fallback
        fallback_runnable = fallback_model.with_fallbacks([])
        primary_model.set_fallback(fallback_runnable)

        # Test with string input
        result = primary_model.invoke("string input")
        assert result.content == "fallback used"

        # Reset for next test
        primary_model.set_fallback(fallback_runnable)

        # Test with message list input
        messages = [HumanMessage(content="test")]
        result = primary_model.invoke(messages)
        assert result.content == "fallback used"

    def test_multiple_fallback_resets(self) -> None:
        """Test that multiple calls to _reset_fallback work correctly."""
        model = FakeChatModelForFallbackTesting(responses=["test"])
        fallback_model = FakeChatModelForFallbackTesting(responses=["fallback"])
        fallback_runnable = fallback_model.with_fallbacks([])

        # Set fallback
        model.set_fallback(fallback_runnable)
        assert model.get_fallback() is not None

        # Reset multiple times
        model._reset_fallback()
        assert model.get_fallback() is None

        model._reset_fallback()  # Should not error
        assert model.get_fallback() is None

        model._reset_fallback()  # Should not error
        assert model.get_fallback() is None
