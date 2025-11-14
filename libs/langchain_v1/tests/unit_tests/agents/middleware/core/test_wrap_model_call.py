"""Unit tests for wrap_model_call hook and @wrap_model_call decorator.

This module tests the wrap_model_call functionality in three forms:
1. As a middleware method (AgentMiddleware.wrap_model_call)
2. As a decorator (@wrap_model_call)
3. Async variant (AgentMiddleware.awrap_model_call)
"""

from collections.abc import Awaitable, Callable

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langchain.agents import create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    wrap_model_call,
)

from ...model import FakeToolCallingModel


class TestBasicWrapModelCall:
    """Test basic wrap_model_call functionality."""

    def test_passthrough_middleware(self) -> None:
        """Test middleware that simply passes through without modification."""

        class PassthroughMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                return handler(request)

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        agent = create_agent(model=model, middleware=[PassthroughMiddleware()])

        result = agent.invoke({"messages": [HumanMessage("Hi")]})

        assert len(result["messages"]) == 2
        assert result["messages"][1].content == "Hello"

    def test_logging_middleware(self) -> None:
        """Test middleware that logs calls without modification."""
        call_log = []

        class LoggingMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                call_log.append("before")
                result = handler(request)
                call_log.append("after")
                return result

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))
        agent = create_agent(model=model, middleware=[LoggingMiddleware()])

        result = agent.invoke({"messages": [HumanMessage("Test")]})

        assert call_log == ["before", "after"]
        assert result["messages"][1].content == "Response"

    def test_counting_middleware(self) -> None:
        """Test middleware that counts model calls."""

        class CountingMiddleware(AgentMiddleware):
            def __init__(self):
                super().__init__()
                self.call_count = 0

            def wrap_model_call(self, request, handler):
                self.call_count += 1
                return handler(request)

        counter = CountingMiddleware()
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Reply")]))
        agent = create_agent(model=model, middleware=[counter])

        agent.invoke({"messages": [HumanMessage("Test")]})

        assert counter.call_count == 1


class TestRetryLogic:
    """Test retry logic with wrap_model_call."""

    def test_simple_retry_on_error(self) -> None:
        """Test middleware that retries once on error."""
        call_count = {"value": 0}

        class FailOnceThenSucceed(GenericFakeChatModel):
            def _generate(self, messages, **kwargs):
                call_count["value"] += 1
                if call_count["value"] == 1:
                    raise ValueError("First call fails")
                return super()._generate(messages, **kwargs)

        class RetryOnceMiddleware(AgentMiddleware):
            def __init__(self):
                super().__init__()
                self.retry_count = 0

            def wrap_model_call(self, request, handler):
                try:
                    return handler(request)
                except Exception:
                    self.retry_count += 1
                    return handler(request)

        retry_middleware = RetryOnceMiddleware()
        model = FailOnceThenSucceed(messages=iter([AIMessage(content="Success")]))
        agent = create_agent(model=model, middleware=[retry_middleware])

        result = agent.invoke({"messages": [HumanMessage("Test")]})

        assert retry_middleware.retry_count == 1
        assert result["messages"][1].content == "Success"

    def test_max_retries(self) -> None:
        """Test middleware with maximum retry limit."""

        class AlwaysFailModel(GenericFakeChatModel):
            def _generate(self, messages, **kwargs):
                raise ValueError("Always fails")

        class MaxRetriesMiddleware(AgentMiddleware):
            def __init__(self, max_retries=3):
                super().__init__()
                self.max_retries = max_retries
                self.attempts = []

            def wrap_model_call(self, request, handler):
                last_exception = None
                for attempt in range(self.max_retries):
                    self.attempts.append(attempt + 1)
                    try:
                        return handler(request)
                    except Exception as e:
                        last_exception = e
                        continue
                # Re-raise the last exception
                if last_exception:
                    raise last_exception

        retry_middleware = MaxRetriesMiddleware(max_retries=3)
        model = AlwaysFailModel(messages=iter([]))
        agent = create_agent(model=model, middleware=[retry_middleware])

        with pytest.raises(ValueError, match="Always fails"):
            agent.invoke({"messages": [HumanMessage("Test")]})

        assert retry_middleware.attempts == [1, 2, 3]

    def test_no_retry_propagates_error(self) -> None:
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

    def test_max_attempts_limit(self) -> None:
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


class TestResponseRewriting:
    """Test response content rewriting with wrap_model_call."""

    def test_uppercase_response(self) -> None:
        """Test middleware that transforms response to uppercase."""

        class UppercaseMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                result = handler(request)
                # result is ModelResponse, extract AIMessage from it
                ai_message = result.result[0]
                return AIMessage(content=ai_message.content.upper())

        model = GenericFakeChatModel(messages=iter([AIMessage(content="hello world")]))
        agent = create_agent(model=model, middleware=[UppercaseMiddleware()])

        result = agent.invoke({"messages": [HumanMessage("Test")]})

        assert result["messages"][1].content == "HELLO WORLD"

    def test_prefix_response(self) -> None:
        """Test middleware that adds prefix to response."""

        class PrefixMiddleware(AgentMiddleware):
            def __init__(self, prefix: str):
                super().__init__()
                self.prefix = prefix

            def wrap_model_call(self, request, handler):
                result = handler(request)
                # result is ModelResponse, extract AIMessage from it
                ai_message = result.result[0]
                return AIMessage(content=f"{self.prefix}{ai_message.content}")

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))
        agent = create_agent(model=model, middleware=[PrefixMiddleware(prefix="[BOT]: ")])

        result = agent.invoke({"messages": [HumanMessage("Test")]})

        assert result["messages"][1].content == "[BOT]: Response"

    def test_multi_stage_transformation(self) -> None:
        """Test middleware applying multiple transformations."""

        class MultiTransformMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                result = handler(request)
                # result is ModelResponse, extract AIMessage from it
                ai_message = result.result[0]

                # First transformation: uppercase
                content = ai_message.content.upper()
                # Second transformation: add prefix and suffix
                content = f"[START] {content} [END]"
                return AIMessage(content=content)

        model = GenericFakeChatModel(messages=iter([AIMessage(content="hello")]))
        agent = create_agent(model=model, middleware=[MultiTransformMiddleware()])

        result = agent.invoke({"messages": [HumanMessage("Test")]})

        assert result["messages"][1].content == "[START] HELLO [END]"


class TestErrorHandling:
    """Test error handling with wrap_model_call."""

    def test_convert_error_to_response(self) -> None:
        """Test middleware that converts errors to successful responses."""

        class AlwaysFailModel(GenericFakeChatModel):
            def _generate(self, messages, **kwargs):
                raise ValueError("Model error")

        class ErrorToSuccessMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                try:
                    return handler(request)
                except Exception as e:
                    return AIMessage(content=f"Error occurred: {e}. Using fallback response.")

        model = AlwaysFailModel(messages=iter([]))
        agent = create_agent(model=model, middleware=[ErrorToSuccessMiddleware()])

        # Should not raise, middleware converts error to response
        result = agent.invoke({"messages": [HumanMessage("Test")]})

        assert "Error occurred" in result["messages"][1].content
        assert "fallback response" in result["messages"][1].content

    def test_selective_error_handling(self) -> None:
        """Test middleware that only handles specific errors."""

        class SpecificErrorModel(GenericFakeChatModel):
            def _generate(self, messages, **kwargs):
                raise ConnectionError("Network error")

        class SelectiveErrorMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                try:
                    return handler(request)
                except ConnectionError:
                    return AIMessage(content="Network issue, try again later")

        model = SpecificErrorModel(messages=iter([]))
        agent = create_agent(model=model, middleware=[SelectiveErrorMiddleware()])

        result = agent.invoke({"messages": [HumanMessage("Test")]})

        assert result["messages"][1].content == "Network issue, try again later"

    def test_error_handling_with_success_path(self) -> None:
        """Test that error handling middleware works correctly on both success and error paths."""
        call_log = []

        class ErrorRecoveryMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                try:
                    call_log.append("before-yield")
                    result = handler(request)
                    call_log.append("after-yield-success")
                    return result
                except Exception:
                    call_log.append("caught-error")
                    return AIMessage(content="Recovered from error")

        # Test 1: Success path
        call_log.clear()
        model1 = GenericFakeChatModel(messages=iter([AIMessage(content="Success")]))
        agent1 = create_agent(model=model1, middleware=[ErrorRecoveryMiddleware()])
        result1 = agent1.invoke({"messages": [HumanMessage("Test")]})

        assert result1["messages"][1].content == "Success"
        assert call_log == ["before-yield", "after-yield-success"]

        # Test 2: Error path
        call_log.clear()

        class AlwaysFailModel(GenericFakeChatModel):
            def _generate(self, messages, **kwargs):
                raise ValueError("Model error")

        model2 = AlwaysFailModel(messages=iter([]))
        agent2 = create_agent(model=model2, middleware=[ErrorRecoveryMiddleware()])
        result2 = agent2.invoke({"messages": [HumanMessage("Test")]})

        assert result2["messages"][1].content == "Recovered from error"
        assert call_log == ["before-yield", "caught-error"]


class TestShortCircuit:
    """Test short-circuit patterns with wrap_model_call."""

    def test_cache_short_circuit(self) -> None:
        """Test middleware that short-circuits with cached response."""
        cache = {}
        model_calls = []

        class CachingMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                # Simple cache key based on last message
                cache_key = str(request.messages[-1].content) if request.messages else ""

                if cache_key in cache:
                    # Short-circuit with cached result
                    return cache[cache_key]
                else:
                    # Execute and cache
                    result = handler(request)
                    cache[cache_key] = result
                    return result

        class TrackingModel(GenericFakeChatModel):
            def _generate(self, messages, **kwargs):
                model_calls.append(len(messages))
                return super()._generate(messages, **kwargs)

        model = TrackingModel(
            messages=iter(
                [
                    AIMessage(content="Response 1"),
                    AIMessage(content="Response 2"),
                ]
            )
        )
        agent = create_agent(model=model, middleware=[CachingMiddleware()])

        # First call - cache miss, calls model
        result1 = agent.invoke({"messages": [HumanMessage("Hello")]})
        assert result1["messages"][1].content == "Response 1"
        assert len(model_calls) == 1

        # Second call with same message - cache hit, doesn't call model
        result2 = agent.invoke({"messages": [HumanMessage("Hello")]})
        assert result2["messages"][1].content == "Response 1"
        assert len(model_calls) == 1  # Still 1, no new call

        # Third call with different message - cache miss, calls model
        result3 = agent.invoke({"messages": [HumanMessage("Goodbye")]})
        assert result3["messages"][1].content == "Response 2"
        assert len(model_calls) == 2  # New call


class TestRequestModification:
    """Test request modification with wrap_model_call."""

    def test_add_system_prompt(self) -> None:
        """Test middleware that adds a system prompt to requests."""
        received_requests = []

        class SystemPromptMiddleware(AgentMiddleware):
            def __init__(self, system_prompt: str):
                super().__init__()
                self.system_prompt = system_prompt

            def wrap_model_call(self, request, handler):
                # Modify request to add system prompt
                modified_request = ModelRequest(
                    model=request.model,
                    system_prompt=self.system_prompt,
                    messages=request.messages,
                    tools=request.tools,
                    tool_choice=request.tool_choice,
                    response_format=request.response_format,
                    model_settings=request.model_settings,
                    state=request.state,
                    runtime=request.runtime,
                )
                received_requests.append(modified_request)
                return handler(modified_request)

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))
        agent = create_agent(
            model=model,
            middleware=[SystemPromptMiddleware(system_prompt="You are a helpful assistant.")],
        )

        result = agent.invoke({"messages": [HumanMessage("Test")]})

        assert len(received_requests) == 1
        assert received_requests[0].system_prompt == "You are a helpful assistant."
        assert result["messages"][1].content == "Response"


class TestStateAndRuntime:
    """Test state and runtime access in wrap_model_call."""

    def test_access_state_in_middleware(self) -> None:
        """Test middleware can read and use state."""
        state_values = []

        class StateAwareMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                # Access state from request
                state_values.append(
                    {
                        "messages_count": len(request.state.get("messages", [])),
                    }
                )
                return handler(request)

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))
        agent = create_agent(model=model, middleware=[StateAwareMiddleware()])

        result = agent.invoke({"messages": [HumanMessage("Test")]})

        assert len(state_values) == 1
        assert state_values[0]["messages_count"] == 1  # Just The HumanMessage
        assert result["messages"][1].content == "Response"

    def test_retry_with_state_tracking(self) -> None:
        """Test middleware that tracks retry count in state."""

        class StateTrackingRetryMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                max_retries = 2
                for attempt in range(max_retries):
                    try:
                        return handler(request)
                    except Exception:
                        if attempt == max_retries - 1:
                            raise

        call_count = {"value": 0}

        class FailOnceThenSucceed(GenericFakeChatModel):
            def _generate(self, messages, **kwargs):
                call_count["value"] += 1
                if call_count["value"] == 1:
                    raise ValueError("First fails")
                return super()._generate(messages, **kwargs)

        model = FailOnceThenSucceed(messages=iter([AIMessage(content="Success")]))
        agent = create_agent(model=model, middleware=[StateTrackingRetryMiddleware()])

        result = agent.invoke({"messages": [HumanMessage("Test")]})

        assert call_count["value"] == 2  # Failed once, succeeded second time
        assert result["messages"][1].content == "Success"


class TestMiddlewareComposition:
    """Test composition of multiple wrap_model_call middleware."""

    def test_two_middleware_composition(self) -> None:
        """Test that two middleware compose correctly (outer wraps inner)."""
        execution_order = []

        class OuterMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                execution_order.append("outer-before")
                response = handler(request)
                execution_order.append("outer-after")
                return response

        class InnerMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                execution_order.append("inner-before")
                response = handler(request)
                execution_order.append("inner-after")
                return response

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))
        agent = create_agent(model=model, middleware=[OuterMiddleware(), InnerMiddleware()])

        agent.invoke({"messages": [HumanMessage("Test")]})

        # Outer wraps inner: outer-before, inner-before, model, inner-after, outer-after
        assert execution_order == [
            "outer-before",
            "inner-before",
            "inner-after",
            "outer-after",
        ]

    def test_three_middleware_composition(self) -> None:
        """Test composition of three middleware."""
        execution_order = []

        class FirstMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                execution_order.append("first-before")
                response = handler(request)
                execution_order.append("first-after")
                return response

        class SecondMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                execution_order.append("second-before")
                response = handler(request)
                execution_order.append("second-after")
                return response

        class ThirdMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                execution_order.append("third-before")
                response = handler(request)
                execution_order.append("third-after")
                return response

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))
        agent = create_agent(
            model=model,
            middleware=[FirstMiddleware(), SecondMiddleware(), ThirdMiddleware()],
        )

        agent.invoke({"messages": [HumanMessage("Test")]})

        # First wraps Second wraps Third: 1-before, 2-before, 3-before, model, 3-after, 2-after, 1-after
        assert execution_order == [
            "first-before",
            "second-before",
            "third-before",
            "third-after",
            "second-after",
            "first-after",
        ]

    def test_retry_with_logging(self) -> None:
        """Test retry middleware composed with logging middleware."""
        call_count = {"value": 0}
        log = []

        class FailOnceThenSucceed(GenericFakeChatModel):
            def _generate(self, messages, **kwargs):
                call_count["value"] += 1
                if call_count["value"] == 1:
                    raise ValueError("First call fails")
                return super()._generate(messages, **kwargs)

        class LoggingMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                log.append("logging-before")
                result = handler(request)
                log.append("logging-after")
                return result

        class RetryMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                log.append("retry-before")
                try:
                    result = handler(request)
                    log.append("retry-after")
                    return result
                except Exception:
                    log.append("retry-retrying")
                    result = handler(request)
                    log.append("retry-after")
                    return result

        model = FailOnceThenSucceed(messages=iter([AIMessage(content="Success")]))
        # Logging is outer, Retry is inner
        agent = create_agent(model=model, middleware=[LoggingMiddleware(), RetryMiddleware()])

        result = agent.invoke({"messages": [HumanMessage("Test")]})

        assert result["messages"][1].content == "Success"
        # Outer (logging) sees the final result after inner (retry) handles it
        assert log == [
            "logging-before",
            "retry-before",
            "retry-retrying",
            "retry-after",
            "logging-after",
        ]

    def test_multiple_transformations(self) -> None:
        """Test multiple middleware that each transform the response."""

        class PrefixMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                result = handler(request)
                # result is ModelResponse, extract AIMessage from it
                ai_message = result.result[0]
                return AIMessage(content=f"[PREFIX] {ai_message.content}")

        class SuffixMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                result = handler(request)
                # result is ModelResponse, extract AIMessage from it
                ai_message = result.result[0]
                return AIMessage(content=f"{ai_message.content} [SUFFIX]")

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Middle")]))
        # Prefix is outer, Suffix is inner
        # Inner (Suffix) runs first, then Outer (Prefix)
        agent = create_agent(model=model, middleware=[PrefixMiddleware(), SuffixMiddleware()])

        result = agent.invoke({"messages": [HumanMessage("Test")]})

        # Suffix adds suffix first, then Prefix adds prefix
        assert result["messages"][1].content == "[PREFIX] Middle [SUFFIX]"

    def test_retry_outer_transform_inner(self) -> None:
        """Test retry as outer middleware with transform as inner."""
        call_count = {"value": 0}

        class FailOnceThenSucceed(GenericFakeChatModel):
            def _generate(self, messages, **kwargs):
                call_count["value"] += 1
                if call_count["value"] == 1:
                    raise ValueError("First call fails")
                return super()._generate(messages, **kwargs)

        class RetryMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                try:
                    return handler(request)
                except Exception:
                    return handler(request)

        class UppercaseMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                result = handler(request)
                # result is ModelResponse, extract AIMessage from it
                ai_message = result.result[0]
                return AIMessage(content=ai_message.content.upper())

        model = FailOnceThenSucceed(messages=iter([AIMessage(content="success")]))
        # Retry outer, Uppercase inner
        agent = create_agent(model=model, middleware=[RetryMiddleware(), UppercaseMiddleware()])

        result = agent.invoke({"messages": [HumanMessage("Test")]})

        # Should retry and uppercase the result
        assert result["messages"][1].content == "SUCCESS"

    def test_middle_retry_middleware(self) -> None:
        """Test that middle middleware doing retry causes inner to execute twice."""
        execution_order = []
        model_calls = []

        class OuterMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                execution_order.append("outer-before")
                result = handler(request)
                execution_order.append("outer-after")
                return result

        class MiddleRetryMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                execution_order.append("middle-before")
                # Always retry once (call handler twice)
                result = handler(request)
                execution_order.append("middle-retry")
                result = handler(request)
                execution_order.append("middle-after")
                return result

        class InnerMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                execution_order.append("inner-before")
                result = handler(request)
                execution_order.append("inner-after")
                return result

        class TrackingModel(GenericFakeChatModel):
            def _generate(self, messages, **kwargs):
                model_calls.append(len(messages))
                return super()._generate(messages, **kwargs)

        model = TrackingModel(
            messages=iter([AIMessage(content="Response 1"), AIMessage(content="Response 2")])
        )
        agent = create_agent(
            model=model,
            middleware=[OuterMiddleware(), MiddleRetryMiddleware(), InnerMiddleware()],
        )

        agent.invoke({"messages": [HumanMessage("Test")]})

        # Middle yields twice, so inner runs twice
        assert execution_order == [
            "outer-before",
            "middle-before",
            "inner-before",  # First execution
            "inner-after",
            "middle-retry",  # Middle yields again
            "inner-before",  # Second execution
            "inner-after",
            "middle-after",
            "outer-after",
        ]
        # Model should be called twice
        assert len(model_calls) == 2


class TestWrapModelCallDecorator:
    """Test the @wrap_model_call decorator for creating middleware."""

    def test_basic_decorator_usage(self) -> None:
        """Test basic decorator usage without parameters."""

        @wrap_model_call
        def passthrough_middleware(request, handler):
            return handler(request)

        # Should return an AgentMiddleware instance
        assert isinstance(passthrough_middleware, AgentMiddleware)

        # Should work in agent
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        agent = create_agent(model=model, middleware=[passthrough_middleware])

        result = agent.invoke({"messages": [HumanMessage("Hi")]})
        assert len(result["messages"]) == 2
        assert result["messages"][1].content == "Hello"

    def test_decorator_with_custom_name(self) -> None:
        """Test decorator with custom middleware name."""

        @wrap_model_call(name="CustomMiddleware")
        def my_middleware(request, handler):
            return handler(request)

        assert isinstance(my_middleware, AgentMiddleware)
        assert my_middleware.__class__.__name__ == "CustomMiddleware"

    def test_decorator_retry_logic(self) -> None:
        """Test decorator for implementing retry logic."""
        call_count = {"value": 0}

        class FailOnceThenSucceed(GenericFakeChatModel):
            def _generate(self, messages, **kwargs):
                call_count["value"] += 1
                if call_count["value"] == 1:
                    raise ValueError("First call fails")
                return super()._generate(messages, **kwargs)

        @wrap_model_call
        def retry_once(request, handler):
            try:
                return handler(request)
            except Exception:
                # Retry once
                return handler(request)

        model = FailOnceThenSucceed(messages=iter([AIMessage(content="Success")]))
        agent = create_agent(model=model, middleware=[retry_once])

        result = agent.invoke({"messages": [HumanMessage("Test")]})

        assert call_count["value"] == 2
        assert result["messages"][1].content == "Success"

    def test_decorator_response_rewriting(self) -> None:
        """Test decorator for rewriting responses."""

        @wrap_model_call
        def uppercase_responses(request, handler):
            result = handler(request)
            # result is ModelResponse, extract AIMessage from it
            ai_message = result.result[0]
            return AIMessage(content=ai_message.content.upper())

        model = GenericFakeChatModel(messages=iter([AIMessage(content="hello world")]))
        agent = create_agent(model=model, middleware=[uppercase_responses])

        result = agent.invoke({"messages": [HumanMessage("Test")]})

        assert result["messages"][1].content == "HELLO WORLD"

    def test_decorator_error_handling(self) -> None:
        """Test decorator for error recovery."""

        class AlwaysFailModel(GenericFakeChatModel):
            def _generate(self, messages, **kwargs):
                raise ValueError("Model error")

        @wrap_model_call
        def error_to_fallback(request, handler):
            try:
                return handler(request)
            except Exception:
                return AIMessage(content="Fallback response")

        model = AlwaysFailModel(messages=iter([]))
        agent = create_agent(model=model, middleware=[error_to_fallback])

        result = agent.invoke({"messages": [HumanMessage("Test")]})

        assert result["messages"][1].content == "Fallback response"

    def test_decorator_with_state_access(self) -> None:
        """Test decorator accessing agent state."""
        state_values = []

        @wrap_model_call
        def log_state(request, handler):
            state_values.append(request.state.get("messages"))
            return handler(request)

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))
        agent = create_agent(model=model, middleware=[log_state])

        agent.invoke({"messages": [HumanMessage("Test")]})

        # State should contain the user message
        assert len(state_values) == 1
        assert len(state_values[0]) == 1
        assert state_values[0][0].content == "Test"

    def test_multiple_decorated_middleware(self) -> None:
        """Test composition of multiple decorated middleware."""
        execution_order = []

        @wrap_model_call
        def outer_middleware(request, handler):
            execution_order.append("outer-before")
            result = handler(request)
            execution_order.append("outer-after")
            return result

        @wrap_model_call
        def inner_middleware(request, handler):
            execution_order.append("inner-before")
            result = handler(request)
            execution_order.append("inner-after")
            return result

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))
        agent = create_agent(model=model, middleware=[outer_middleware, inner_middleware])

        agent.invoke({"messages": [HumanMessage("Test")]})

        assert execution_order == [
            "outer-before",
            "inner-before",
            "inner-after",
            "outer-after",
        ]

    def test_decorator_with_custom_state_schema(self) -> None:
        """Test decorator with custom state schema."""
        from typing_extensions import TypedDict

        class CustomState(TypedDict):
            messages: list
            custom_field: str

        @wrap_model_call(state_schema=CustomState)
        def middleware_with_schema(request, handler):
            return handler(request)

        assert isinstance(middleware_with_schema, AgentMiddleware)
        # Custom state schema should be set
        assert middleware_with_schema.state_schema == CustomState

    def test_decorator_with_tools_parameter(self) -> None:
        """Test decorator with tools parameter."""
        from langchain_core.tools import tool

        @tool
        def test_tool(query: str) -> str:
            """A test tool."""
            return f"Result: {query}"

        @wrap_model_call(tools=[test_tool])
        def middleware_with_tools(request, handler):
            return handler(request)

        assert isinstance(middleware_with_tools, AgentMiddleware)
        assert len(middleware_with_tools.tools) == 1
        assert middleware_with_tools.tools[0].name == "test_tool"

    def test_decorator_parentheses_optional(self) -> None:
        """Test that decorator works both with and without parentheses."""

        # Without parentheses
        @wrap_model_call
        def middleware_no_parens(request, handler):
            return handler(request)

        # With parentheses
        @wrap_model_call()
        def middleware_with_parens(request, handler):
            return handler(request)

        assert isinstance(middleware_no_parens, AgentMiddleware)
        assert isinstance(middleware_with_parens, AgentMiddleware)

    def test_decorator_preserves_function_name(self) -> None:
        """Test that decorator uses function name for class name."""

        @wrap_model_call
        def my_custom_middleware(request, handler):
            return handler(request)

        assert my_custom_middleware.__class__.__name__ == "my_custom_middleware"

    def test_decorator_mixed_with_class_middleware(self) -> None:
        """Test decorated middleware mixed with class-based middleware."""
        execution_order = []

        @wrap_model_call
        def decorated_middleware(request, handler):
            execution_order.append("decorated-before")
            result = handler(request)
            execution_order.append("decorated-after")
            return result

        class ClassMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                execution_order.append("class-before")
                result = handler(request)
                execution_order.append("class-after")
                return result

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))
        agent = create_agent(
            model=model,
            middleware=[decorated_middleware, ClassMiddleware()],
        )

        agent.invoke({"messages": [HumanMessage("Test")]})

        # Decorated is outer, class-based is inner
        assert execution_order == [
            "decorated-before",
            "class-before",
            "class-after",
            "decorated-after",
        ]

    def test_decorator_complex_retry_logic(self) -> None:
        """Test decorator with complex retry logic and backoff."""
        attempts = []
        call_count = {"value": 0}

        class UnreliableModel(GenericFakeChatModel):
            def _generate(self, messages, **kwargs):
                call_count["value"] += 1
                if call_count["value"] <= 2:
                    raise ValueError(f"Attempt {call_count['value']} failed")
                return super()._generate(messages, **kwargs)

        @wrap_model_call
        def retry_with_tracking(request, handler):
            max_retries = 3
            for attempt in range(max_retries):
                attempts.append(attempt + 1)
                try:
                    return handler(request)
                except Exception:
                    # On error, continue to next attempt
                    if attempt < max_retries - 1:
                        continue  # Retry
                    else:
                        raise  # All retries failed

        model = UnreliableModel(messages=iter([AIMessage(content="Finally worked")]))
        agent = create_agent(model=model, middleware=[retry_with_tracking])

        result = agent.invoke({"messages": [HumanMessage("Test")]})

        assert attempts == [1, 2, 3]
        assert result["messages"][1].content == "Finally worked"

    def test_decorator_request_modification(self) -> None:
        """Test decorator modifying request before execution."""
        modified_prompts = []

        @wrap_model_call
        def add_system_prompt(request, handler):
            # Modify request to add system prompt
            modified_request = ModelRequest(
                messages=request.messages,
                model=request.model,
                system_prompt="You are a helpful assistant",
                tool_choice=request.tool_choice,
                tools=request.tools,
                response_format=request.response_format,
                state={},
                runtime=None,
            )
            modified_prompts.append(modified_request.system_prompt)
            return handler(modified_request)

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))
        agent = create_agent(model=model, middleware=[add_system_prompt])

        agent.invoke({"messages": [HumanMessage("Test")]})

        assert modified_prompts == ["You are a helpful assistant"]


class TestAsyncWrapModelCall:
    """Test async execution with wrap_model_call."""

    async def test_async_model_with_middleware(self) -> None:
        """Test that wrap_model_call works with async model execution."""
        log = []

        class LoggingMiddleware(AgentMiddleware):
            async def awrap_model_call(self, request, handler):
                log.append("before")
                result = await handler(request)
                log.append("after")
                return result

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Async response")]))
        agent = create_agent(model=model, middleware=[LoggingMiddleware()])

        result = await agent.ainvoke({"messages": [HumanMessage("Test")]})

        assert log == ["before", "after"]
        assert result["messages"][1].content == "Async response"

    async def test_async_retry(self) -> None:
        """Test retry logic with async execution."""
        call_count = {"value": 0}

        class AsyncFailOnceThenSucceed(GenericFakeChatModel):
            async def _agenerate(self, messages, **kwargs):
                call_count["value"] += 1
                if call_count["value"] == 1:
                    raise ValueError("First async call fails")
                return await super()._agenerate(messages, **kwargs)

        class RetryMiddleware(AgentMiddleware):
            async def awrap_model_call(self, request, handler):
                try:
                    return await handler(request)
                except Exception:
                    return await handler(request)

        model = AsyncFailOnceThenSucceed(messages=iter([AIMessage(content="Async success")]))
        agent = create_agent(model=model, middleware=[RetryMiddleware()])

        result = await agent.ainvoke({"messages": [HumanMessage("Test")]})

        assert call_count["value"] == 2
        assert result["messages"][1].content == "Async success"

    async def test_decorator_with_async_agent(self) -> None:
        """Test that decorated middleware works with async agent invocation."""
        call_log = []

        @wrap_model_call
        async def logging_middleware(request, handler):
            call_log.append("before")
            result = await handler(request)
            call_log.append("after")
            return result

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Async response")]))
        agent = create_agent(model=model, middleware=[logging_middleware])

        result = await agent.ainvoke({"messages": [HumanMessage("Test")]})

        assert call_log == ["before", "after"]
        assert result["messages"][1].content == "Async response"


class TestSyncAsyncInterop:
    """Test sync/async interoperability."""

    def test_sync_invoke_with_only_async_middleware_raises_error(self) -> None:
        """Test that sync invoke with only async middleware raises error."""

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

    def test_sync_invoke_with_mixed_middleware(self) -> None:
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


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_middleware_modifies_request(self) -> None:
        """Test middleware that modifies the request before execution."""
        modified_messages = []

        class RequestModifyingMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                # Add a system message to the request
                modified_request = request
                modified_messages.append(len(modified_request.messages))
                return handler(modified_request)

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))
        agent = create_agent(model=model, middleware=[RequestModifyingMiddleware()])

        agent.invoke({"messages": [HumanMessage("Test")]})

        assert len(modified_messages) == 1

    def test_multiple_yields_retry_different_models(self) -> None:
        """Test middleware that tries multiple different models."""
        attempts = []

        class MultiModelRetryMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                attempts.append("first-attempt")
                try:
                    return handler(request)
                except Exception:
                    attempts.append("retry-attempt")
                    return handler(request)

        call_count = {"value": 0}

        class FailFirstSucceedSecond(GenericFakeChatModel):
            def _generate(self, messages, **kwargs):
                call_count["value"] += 1
                if call_count["value"] == 1:
                    raise ValueError("First fails")
                return super()._generate(messages, **kwargs)

        model = FailFirstSucceedSecond(messages=iter([AIMessage(content="Success")]))
        agent = create_agent(model=model, middleware=[MultiModelRetryMiddleware()])

        result = agent.invoke({"messages": [HumanMessage("Test")]})

        assert attempts == ["first-attempt", "retry-attempt"]
        assert result["messages"][1].content == "Success"
