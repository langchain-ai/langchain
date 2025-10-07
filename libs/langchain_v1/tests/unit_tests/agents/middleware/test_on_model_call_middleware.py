"""Unit tests for on_model_call middleware generator protocol."""

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from langchain.agents import create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)


class TestBasicOnModelCall:
    """Test basic on_model_call functionality."""

    def test_passthrough_middleware(self) -> None:
        """Test middleware that simply passes through without modification."""

        class PassthroughMiddleware(AgentMiddleware):
            def on_model_call(self, request, state, runtime):
                response = yield request
                return response

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
                response = yield request
                call_log.append(f"after-{response.action}")
                return response

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))
        agent = create_agent(model=model, middleware=[LoggingMiddleware()])

        result = agent.invoke({"messages": [HumanMessage("Test")]})

        assert call_log == ["before", "after-return"]
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
                return response

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
                response = yield request
                if response.action == "raise":
                    self.retry_count += 1
                    response = yield request
                return response

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
                for attempt in range(self.max_retries):
                    self.attempts.append(attempt + 1)
                    response = yield request
                    if response.action == "return":
                        return response
                return response

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
                response = yield request
                if response.action == "return" and response.result:
                    modified = AIMessage(content=response.result.content.upper())
                    response = ModelResponse(action="return", result=modified)
                return response

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
                response = yield request
                if response.action == "return" and response.result:
                    modified = AIMessage(content=f"{self.prefix}{response.result.content}")
                    response = ModelResponse(action="return", result=modified)
                return response

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
                response = yield request
                if response.action == "raise":
                    fallback = AIMessage(content="Error handled gracefully")
                    response = ModelResponse(action="return", result=fallback)
                return response

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
                response = yield request
                if response.action == "raise":
                    if isinstance(response.exception, ConnectionError):
                        fallback = AIMessage(content="Network issue, try again later")
                        response = ModelResponse(action="return", result=fallback)
                return response

        model = SpecificErrorModel(messages=iter([]))
        agent = create_agent(model=model, middleware=[SelectiveErrorMiddleware()])

        result = agent.invoke({"messages": [HumanMessage("Test")]})

        assert result["messages"][1].content == "Network issue, try again later"


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
                return response

        class InnerMiddleware(AgentMiddleware):
            def on_model_call(self, request, state, runtime):
                execution_order.append("inner-before")
                response = yield request
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
            def on_model_call(self, request, state, runtime):
                log.append("logging-before")
                response = yield request
                log.append(f"logging-after-{response.action}")
                return response

        class RetryMiddleware(AgentMiddleware):
            def on_model_call(self, request, state, runtime):
                log.append("retry-before")
                response = yield request
                if response.action == "raise":
                    log.append("retry-retrying")
                    response = yield request
                log.append(f"retry-after-{response.action}")
                return response

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
            "retry-after-return",
            "logging-after-return",
        ]

    def test_multiple_transformations(self) -> None:
        """Test multiple middleware that each transform the response."""

        class PrefixMiddleware(AgentMiddleware):
            def on_model_call(self, request, state, runtime):
                response = yield request
                if response.action == "return" and response.result:
                    modified = AIMessage(content=f"[PREFIX] {response.result.content}")
                    response = ModelResponse(action="return", result=modified)
                return response

        class SuffixMiddleware(AgentMiddleware):
            def on_model_call(self, request, state, runtime):
                response = yield request
                if response.action == "return" and response.result:
                    modified = AIMessage(content=f"{response.result.content} [SUFFIX]")
                    response = ModelResponse(action="return", result=modified)
                return response

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
                response = yield request
                if response.action == "raise":
                    response = yield request
                return response

        class UppercaseMiddleware(AgentMiddleware):
            def on_model_call(self, request, state, runtime):
                response = yield request
                if response.action == "return" and response.result:
                    modified = AIMessage(content=response.result.content.upper())
                    response = ModelResponse(action="return", result=modified)
                return response

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
                return response

        class SecondMiddleware(AgentMiddleware):
            def on_model_call(self, request, state, runtime):
                execution_order.append("second-before")
                response = yield request
                execution_order.append("second-after")
                return response

        class ThirdMiddleware(AgentMiddleware):
            def on_model_call(self, request, state, runtime):
                execution_order.append("third-before")
                response = yield request
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
                return response

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
                response = yield request
                if response.action == "raise":
                    response = yield request
                return response

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
                return response

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
                response = yield request

                if response.action == "raise":
                    attempts.append("retry-attempt")
                    response = yield request

                return response

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
