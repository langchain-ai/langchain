from collections.abc import Awaitable, Callable

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langchain.agents.factory import create_agent
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest

from ..model import FakeToolCallingModel


def test_wrap_model_call_hook() -> None:
    """Test that wrap_model_call hook is called on model errors."""
    call_count = {"value": 0}

    class FailingModel(BaseChatModel):
        """Model that fails on first call, succeeds on second."""

        def _generate(self, messages, **kwargs):
            call_count["value"] += 1
            if call_count["value"] == 1:
                raise ValueError("First call fails")
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="Success on retry"))]
            )

        @property
        def _llm_type(self):
            return "failing"

    class RetryMiddleware(AgentMiddleware):
        def __init__(self):
            super().__init__()
            self.retry_count = 0

        def wrap_model_call(self, request, handler):
            try:
                return handler(request)
            except Exception:
                # Retry on error
                self.retry_count += 1
                return handler(request)

    failing_model = FailingModel()
    retry_middleware = RetryMiddleware()

    agent = create_agent(model=failing_model, middleware=[retry_middleware])

    result = agent.invoke({"messages": [HumanMessage("Test")]})

    # Should have retried once
    assert retry_middleware.retry_count == 1
    # Should have succeeded on second attempt
    assert len(result["messages"]) == 2
    assert result["messages"][1].content == "Success on retry"


def test_wrap_model_call_retry_count() -> None:
    """Test that wrap_model_call can retry multiple times."""

    class AlwaysFailingModel(BaseChatModel):
        """Model that always fails."""

        def _generate(self, messages, **kwargs):
            raise ValueError("Always fails")

        @property
        def _llm_type(self):
            return "always_failing"

    class AttemptTrackingMiddleware(AgentMiddleware):
        def __init__(self):
            super().__init__()
            self.attempts = []

        def wrap_model_call(self, request, handler):
            max_retries = 3
            last_exception = None
            for attempt in range(max_retries):
                self.attempts.append(attempt + 1)
                try:
                    return handler(request)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        continue  # Retry

            # All retries failed, re-raise the last exception
            if last_exception:
                raise last_exception

    model = AlwaysFailingModel()
    tracker = AttemptTrackingMiddleware()

    agent = create_agent(model=model, middleware=[tracker])

    with pytest.raises(ValueError, match="Always fails"):
        agent.invoke({"messages": [HumanMessage("Test")]})

    # Should have attempted 3 times
    assert tracker.attempts == [1, 2, 3]


def test_wrap_model_call_no_retry() -> None:
    """Test that error is propagated when middleware doesn't retry."""

    class FailingModel(BaseChatModel):
        """Model that always fails."""

        def _generate(self, messages, **kwargs):
            raise ValueError("Model error")

        @property
        def _llm_type(self):
            return "failing"

    class NoRetryMiddleware(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            return handler(request)

    agent = create_agent(model=FailingModel(), middleware=[NoRetryMiddleware()])

    with pytest.raises(ValueError, match="Model error"):
        agent.invoke({"messages": [HumanMessage("Test")]})


def test_wrap_model_call_max_attempts() -> None:
    """Test that middleware controls termination via retry limits."""

    class AlwaysFailingModel(BaseChatModel):
        """Model that always fails."""

        def _generate(self, messages, **kwargs):
            raise ValueError("Always fails")

        @property
        def _llm_type(self):
            return "always_failing"

    class LimitedRetryMiddleware(AgentMiddleware):
        """Middleware that limits its own retries."""

        def __init__(self, max_retries: int = 10):
            super().__init__()
            self.max_retries = max_retries
            self.attempt_count = 0

        def wrap_model_call(self, request, handler):
            last_exception = None
            for attempt in range(self.max_retries):
                self.attempt_count += 1
                try:
                    return handler(request)
                except Exception as e:
                    last_exception = e
                    # Continue to retry

            # All retries exhausted, re-raise the last error
            if last_exception:
                raise last_exception

    model = AlwaysFailingModel()
    middleware = LimitedRetryMiddleware(max_retries=10)

    agent = create_agent(model=model, middleware=[middleware])

    # Should fail with the model's error after middleware stops retrying
    with pytest.raises(ValueError, match="Always fails"):
        agent.invoke({"messages": [HumanMessage("Test")]})

    # Should have attempted exactly 10 times as configured
    assert middleware.attempt_count == 10


async def test_wrap_model_call_async() -> None:
    """Test wrap_model_call hook with async model execution."""
    call_count = {"value": 0}

    class AsyncFailingModel(BaseChatModel):
        """Model that fails on first async call, succeeds on second."""

        def _generate(self, messages, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="sync"))])

        async def _agenerate(self, messages, **kwargs):
            call_count["value"] += 1
            if call_count["value"] == 1:
                raise ValueError("First async call fails")
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="Async retry success"))]
            )

        @property
        def _llm_type(self):
            return "async_failing"

    class AsyncRetryMiddleware(AgentMiddleware):
        def __init__(self):
            super().__init__()
            self.retry_count = 0

        async def awrap_model_call(self, request, handler):
            try:
                return await handler(request)
            except Exception:
                # Retry on error
                self.retry_count += 1
                return await handler(request)

    failing_model = AsyncFailingModel()
    retry_middleware = AsyncRetryMiddleware()

    agent = create_agent(model=failing_model, middleware=[retry_middleware])

    result = await agent.ainvoke({"messages": [HumanMessage("Test")]})

    # Should have retried once
    assert retry_middleware.retry_count == 1
    # Should have succeeded on second attempt
    assert len(result["messages"]) == 2
    assert result["messages"][1].content == "Async retry success"


def test_wrap_model_call_rewrite_response() -> None:
    """Test that middleware can rewrite model responses."""

    class SimpleModel(BaseChatModel):
        """Model that returns a simple response."""

        def _generate(self, messages, **kwargs):
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="Original response"))]
            )

        @property
        def _llm_type(self):
            return "simple"

    class ResponseRewriteMiddleware(AgentMiddleware):
        """Middleware that rewrites the response."""

        def wrap_model_call(self, request, handler):
            result = handler(request)

            # result is ModelResponse, extract AIMessage from it
            ai_message = result.result[0]
            # Rewrite the response
            return AIMessage(content=f"REWRITTEN: {ai_message.content}")

    model = SimpleModel()
    middleware = ResponseRewriteMiddleware()

    agent = create_agent(model=model, middleware=[middleware])

    result = agent.invoke({"messages": [HumanMessage("Test")]})

    # Response should be rewritten by middleware
    assert result["messages"][1].content == "REWRITTEN: Original response"


def test_wrap_model_call_convert_error_to_response() -> None:
    """Test that middleware can convert errors to successful responses."""

    class AlwaysFailingModel(BaseChatModel):
        """Model that always fails."""

        def _generate(self, messages, **kwargs):
            raise ValueError("Model error")

        @property
        def _llm_type(self):
            return "failing"

    class ErrorToResponseMiddleware(AgentMiddleware):
        """Middleware that converts errors to success responses."""

        def wrap_model_call(self, request, handler):
            try:
                return handler(request)
            except Exception as e:
                # Convert error to success response
                return AIMessage(content=f"Error occurred: {e}. Using fallback response.")

    model = AlwaysFailingModel()
    middleware = ErrorToResponseMiddleware()

    agent = create_agent(model=model, middleware=[middleware])

    # Should not raise, middleware converts error to response
    result = agent.invoke({"messages": [HumanMessage("Test")]})

    # Response should be the fallback from middleware
    assert "Error occurred" in result["messages"][1].content
    assert "fallback response" in result["messages"][1].content


def test_create_agent_sync_invoke_with_only_async_middleware_raises_error() -> None:
    """Test that sync invoke with only async middleware works via run_in_executor."""

    class AsyncOnlyMiddleware(AgentMiddleware):
        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[AIMessage]],
        ) -> AIMessage:
            return await handler(request)

    agent = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[AsyncOnlyMiddleware()],
    )

    with pytest.raises(NotImplementedError):
        agent.invoke({"messages": [HumanMessage("hello")]})


def test_create_agent_sync_invoke_with_mixed_middleware() -> None:
    """Test that sync invoke works with mixed sync/async middleware when sync versions exist."""
    calls = []

    class MixedMiddleware(AgentMiddleware):
        def before_model(self, state, runtime) -> None:
            calls.append("MixedMiddleware.before_model")

        async def abefore_model(self, state, runtime) -> None:
            calls.append("MixedMiddleware.abefore_model")

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], AIMessage],
        ) -> AIMessage:
            calls.append("MixedMiddleware.wrap_model_call")
            return handler(request)

        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[AIMessage]],
        ) -> AIMessage:
            calls.append("MixedMiddleware.awrap_model_call")
            return await handler(request)

    agent = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[MixedMiddleware()],
    )

    result = agent.invoke({"messages": [HumanMessage("hello")]})

    # In sync mode, only sync methods should be called
    assert calls == [
        "MixedMiddleware.before_model",
        "MixedMiddleware.wrap_model_call",
    ]
