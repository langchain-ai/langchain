"""Unit tests for wrap_model_call middleware generator protocol."""

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from langchain.agents import create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
)


class TestBasicOnModelCall:
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


class TestRetryMiddleware:
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
                    result = handler(request)
                    return result
                except Exception:
                    self.retry_count += 1
                    result = handler(request)
                    return result

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
                        result = handler(request)
                        return result
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


class TestResponseRewriting:
    """Test response content rewriting with wrap_model_call."""

    def test_uppercase_response(self) -> None:
        """Test middleware that transforms response to uppercase."""

        class UppercaseMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                result = handler(request)
                return AIMessage(content=result.content.upper())

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
                return AIMessage(content=f"{self.prefix}{result.content}")

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))
        agent = create_agent(model=model, middleware=[PrefixMiddleware(prefix="[BOT]: ")])

        result = agent.invoke({"messages": [HumanMessage("Test")]})

        assert result["messages"][1].content == "[BOT]: Response"


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
                except Exception:
                    fallback = AIMessage(content="Error handled gracefully")
                    return fallback

        model = AlwaysFailModel(messages=iter([]))
        agent = create_agent(model=model, middleware=[ErrorToSuccessMiddleware()])

        # Should not raise, middleware converts error to response
        result = agent.invoke({"messages": [HumanMessage("Test")]})

        assert "Error handled gracefully" in result["messages"][1].content

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
                    fallback = AIMessage(content="Network issue, try again later")
                    return fallback

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
                    fallback = AIMessage(content="Recovered from error")
                    return fallback

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
                        "thread_model_call_count": request.state.get("thread_model_call_count", 0),
                        "messages_count": len(request.state.get("messages", [])),
                    }
                )
                return handler(request)

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))
        agent = create_agent(model=model, middleware=[StateAwareMiddleware()])

        result = agent.invoke({"messages": [HumanMessage("Test")]})

        assert len(state_values) == 1
        assert state_values[0]["messages_count"] == 1  # Just the human message
        assert result["messages"][1].content == "Response"

    def test_retry_with_state_tracking(self) -> None:
        """Test middleware that tracks retry count in state."""

        class StateTrackingRetryMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                max_retries = 2
                for attempt in range(max_retries):
                    try:
                        return handler(request)
                        break  # Success
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
                return AIMessage(content=f"[PREFIX] {result.content}")

        class SuffixMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                result = handler(request)
                return AIMessage(content=f"{result.content} [SUFFIX]")

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
                    result = handler(request)
                    return result
                except Exception:
                    result = handler(request)
                    return result

        class UppercaseMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                result = handler(request)
                return AIMessage(content=result.content.upper())

        model = FailOnceThenSucceed(messages=iter([AIMessage(content="success")]))
        # Retry outer, Uppercase inner
        agent = create_agent(model=model, middleware=[RetryMiddleware(), UppercaseMiddleware()])

        result = agent.invoke({"messages": [HumanMessage("Test")]})

        # Should retry and uppercase the result
        assert result["messages"][1].content == "SUCCESS"

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


class TestAsyncOnModelCall:
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
                    result = handler(request)
                    return result
                except Exception:
                    attempts.append("retry-attempt")
                    result = handler(request)
                    return result

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
