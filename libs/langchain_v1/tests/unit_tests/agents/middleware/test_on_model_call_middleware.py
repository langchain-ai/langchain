"""Unit tests for on_model_call middleware generator protocol."""

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
    """Test basic on_model_call functionality."""

    def test_passthrough_middleware(self) -> None:
        """Test middleware that simply passes through without modification."""

        class PassthroughMiddleware(AgentMiddleware):
            def on_model_call(self, request, state, runtime):
                response = yield request
                # Generator ends

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        agent = create_agent(model=model, middleware=[PassthroughMiddleware()])

        result = agent.invoke({"messages": [HumanMessage("Hi")]})

        assert len(result["messages"]) == 2
        assert result["messages"][1].content == "Hello"

    def test_logging_middleware(self) -> None:
        """Test middleware that logs calls without modification."""
        call_log = []

        class LoggingMiddleware(AgentMiddleware):
            def on_model_call(self, request, state, runtime):
                call_log.append("before")
                result = yield request
                call_log.append("after")
                # Generator ends

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

            def on_model_call(self, request, state, runtime):
                self.call_count += 1
                response = yield request
                # Generator ends

        counter = CountingMiddleware()
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Reply")]))
        agent = create_agent(model=model, middleware=[counter])

        agent.invoke({"messages": [HumanMessage("Test")]})

        assert counter.call_count == 1


class TestRetryMiddleware:
    """Test retry logic with on_model_call."""

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

            def on_model_call(self, request, state, runtime):
                try:
                    result = yield request
                    # Generator ends
                except Exception:
                    self.retry_count += 1
                    result = yield request
                    # Generator ends

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

            def on_model_call(self, request, state, runtime):
                last_exception = None
                for attempt in range(self.max_retries):
                    self.attempts.append(attempt + 1)
                    try:
                        result = yield request
                        # Generator ends
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
    """Test response content rewriting with on_model_call."""

    def test_uppercase_response(self) -> None:
        """Test middleware that transforms response to uppercase."""

        class UppercaseMiddleware(AgentMiddleware):
            def on_model_call(self, request, state, runtime):
                result = yield request
                modified = AIMessage(content=result.content.upper())
                yield modified

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

            def on_model_call(self, request, state, runtime):
                result = yield request
                modified = AIMessage(content=f"{self.prefix}{result.content}")
                yield modified

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))
        agent = create_agent(model=model, middleware=[PrefixMiddleware(prefix="[BOT]: ")])

        result = agent.invoke({"messages": [HumanMessage("Test")]})

        assert result["messages"][1].content == "[BOT]: Response"


class TestErrorHandling:
    """Test error handling with on_model_call."""

    def test_convert_error_to_response(self) -> None:
        """Test middleware that converts errors to successful responses."""

        class AlwaysFailModel(GenericFakeChatModel):
            def _generate(self, messages, **kwargs):
                raise ValueError("Model error")

        class ErrorToSuccessMiddleware(AgentMiddleware):
            def on_model_call(self, request, state, runtime):
                try:
                    yield request
                except Exception:
                    fallback = AIMessage(content="Error handled gracefully")
                    yield fallback

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
            def on_model_call(self, request, state, runtime):
                try:
                    yield request
                except ConnectionError:
                    fallback = AIMessage(content="Network issue, try again later")
                    yield fallback

        model = SpecificErrorModel(messages=iter([]))
        agent = create_agent(model=model, middleware=[SelectiveErrorMiddleware()])

        result = agent.invoke({"messages": [HumanMessage("Test")]})

        assert result["messages"][1].content == "Network issue, try again later"

    def test_error_handling_with_success_path(self) -> None:
        """Test that error handling middleware works correctly on both success and error paths."""
        call_log = []

        class ErrorRecoveryMiddleware(AgentMiddleware):
            def on_model_call(self, request, state, runtime):
                try:
                    call_log.append("before-yield")
                    yield request
                    call_log.append("after-yield-success")
                except Exception:
                    call_log.append("caught-error")
                    fallback = AIMessage(content="Recovered from error")
                    yield fallback

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
    """Test short-circuit patterns with on_model_call."""

    def test_cache_short_circuit(self) -> None:
        """Test middleware that short-circuits with cached response."""
        cache = {}
        model_calls = []

        class CachingMiddleware(AgentMiddleware):
            def on_model_call(self, request, state, runtime):
                # Simple cache key based on last message
                cache_key = str(request.messages[-1].content) if request.messages else ""

                if cache_key in cache:
                    # Short-circuit with cached result
                    yield cache[cache_key]
                else:
                    # Execute and cache
                    result = yield request
                    cache[cache_key] = result

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
    """Test request modification with on_model_call."""

    def test_add_system_prompt(self) -> None:
        """Test middleware that adds a system prompt to requests."""
        received_requests = []

        class SystemPromptMiddleware(AgentMiddleware):
            def __init__(self, system_prompt: str):
                super().__init__()
                self.system_prompt = system_prompt

            def on_model_call(self, request, state, runtime):
                # Modify request to add system prompt
                modified_request = ModelRequest(
                    model=request.model,
                    system_prompt=self.system_prompt,
                    messages=request.messages,
                    tools=request.tools,
                    tool_choice=request.tool_choice,
                    response_format=request.response_format,
                    model_settings=request.model_settings,
                )
                received_requests.append(modified_request)
                yield modified_request

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
    """Test state and runtime access in on_model_call."""

    def test_access_state_in_middleware(self) -> None:
        """Test middleware can read and use state."""
        state_values = []

        class StateAwareMiddleware(AgentMiddleware):
            def on_model_call(self, request, state, runtime):
                # Access state
                state_values.append(
                    {
                        "thread_model_call_count": state.get("thread_model_call_count", 0),
                        "messages_count": len(state.get("messages", [])),
                    }
                )
                yield request

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))
        agent = create_agent(model=model, middleware=[StateAwareMiddleware()])

        result = agent.invoke({"messages": [HumanMessage("Test")]})

        assert len(state_values) == 1
        assert state_values[0]["messages_count"] == 1  # Just the human message
        assert result["messages"][1].content == "Response"

    def test_retry_with_state_tracking(self) -> None:
        """Test middleware that tracks retry count in state."""

        class StateTrackingRetryMiddleware(AgentMiddleware):
            def on_model_call(self, request, state, runtime):
                max_retries = 2
                for attempt in range(max_retries):
                    try:
                        yield request
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
    """Test composition of multiple on_model_call middleware."""

    def test_two_middleware_composition(self) -> None:
        """Test that two middleware compose correctly (outer wraps inner)."""
        execution_order = []

        class OuterMiddleware(AgentMiddleware):
            def on_model_call(self, request, state, runtime):
                execution_order.append("outer-before")
                response = yield request
                execution_order.append("outer-after")
                # Generator ends

        class InnerMiddleware(AgentMiddleware):
            def on_model_call(self, request, state, runtime):
                execution_order.append("inner-before")
                response = yield request
                execution_order.append("inner-after")
                # Generator ends

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
            def on_model_call(self, request, state, runtime):
                log.append("logging-before")
                result = yield request
                log.append("logging-after")
                # Generator ends

        class RetryMiddleware(AgentMiddleware):
            def on_model_call(self, request, state, runtime):
                log.append("retry-before")
                try:
                    result = yield request
                    log.append("retry-after")
                    # Generator ends
                except Exception:
                    log.append("retry-retrying")
                    result = yield request
                    log.append("retry-after")
                    # Generator ends

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
            def on_model_call(self, request, state, runtime):
                result = yield request
                modified = AIMessage(content=f"[PREFIX] {result.content}")
                yield modified

        class SuffixMiddleware(AgentMiddleware):
            def on_model_call(self, request, state, runtime):
                result = yield request
                modified = AIMessage(content=f"{result.content} [SUFFIX]")
                yield modified

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
            def on_model_call(self, request, state, runtime):
                try:
                    result = yield request
                    # Generator ends
                except Exception:
                    result = yield request
                    # Generator ends

        class UppercaseMiddleware(AgentMiddleware):
            def on_model_call(self, request, state, runtime):
                result = yield request
                modified = AIMessage(content=result.content.upper())
                yield modified

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
            def on_model_call(self, request, state, runtime):
                execution_order.append("first-before")
                response = yield request
                execution_order.append("first-after")
                # Generator ends

        class SecondMiddleware(AgentMiddleware):
            def on_model_call(self, request, state, runtime):
                execution_order.append("second-before")
                response = yield request
                execution_order.append("second-after")
                # Generator ends

        class ThirdMiddleware(AgentMiddleware):
            def on_model_call(self, request, state, runtime):
                execution_order.append("third-before")
                response = yield request
                execution_order.append("third-after")
                # Generator ends

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
            def on_model_call(self, request, state, runtime):
                execution_order.append("outer-before")
                yield request
                execution_order.append("outer-after")

        class MiddleRetryMiddleware(AgentMiddleware):
            def on_model_call(self, request, state, runtime):
                execution_order.append("middle-before")
                # Always retry once (yield twice)
                yield request
                execution_order.append("middle-retry")
                yield request
                execution_order.append("middle-after")

        class InnerMiddleware(AgentMiddleware):
            def on_model_call(self, request, state, runtime):
                execution_order.append("inner-before")
                yield request
                execution_order.append("inner-after")

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
    """Test async execution with on_model_call."""

    async def test_async_model_with_middleware(self) -> None:
        """Test that on_model_call works with async model execution."""
        log = []

        class LoggingMiddleware(AgentMiddleware):
            def on_model_call(self, request, state, runtime):
                log.append("before")
                response = yield request
                log.append("after")
                # Generator ends

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
            def on_model_call(self, request, state, runtime):
                try:
                    result = yield request
                    # Generator ends
                except Exception:
                    result = yield request
                    # Generator ends

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
            def on_model_call(self, request, state, runtime):
                # Add a system message to the request
                modified_request = request
                modified_messages.append(len(modified_request.messages))
                response = yield modified_request
                # Generator ends

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))
        agent = create_agent(model=model, middleware=[RequestModifyingMiddleware()])

        agent.invoke({"messages": [HumanMessage("Test")]})

        assert len(modified_messages) == 1

    def test_multiple_yields_retry_different_models(self) -> None:
        """Test middleware that tries multiple different models."""
        attempts = []

        class MultiModelRetryMiddleware(AgentMiddleware):
            def on_model_call(self, request, state, runtime):
                attempts.append("first-attempt")
                try:
                    result = yield request
                    # Generator ends
                except Exception:
                    attempts.append("retry-attempt")
                    result = yield request
                    # Generator ends

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
