"""Unit tests for the @wrap_model_call decorator."""

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from langchain.agents import create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelCall,
    ModelRequest,
    ModelResponse,
    wrap_model_call,
)


class TestOnModelCallDecorator:
    """Test the @wrap_model_call decorator for creating middleware."""

    def test_basic_decorator_usage(self) -> None:
        """Test basic decorator usage without parameters."""

        @wrap_model_call
        def passthrough_middleware(request, handler):
            return handler(request.model_call)

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
            return handler(request.model_call)

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
                return handler(request.model_call)
            except Exception:
                # Retry once
                return handler(request.model_call)

        model = FailOnceThenSucceed(messages=iter([AIMessage(content="Success")]))
        agent = create_agent(model=model, middleware=[retry_once])

        result = agent.invoke({"messages": [HumanMessage("Test")]})

        assert call_count["value"] == 2
        assert result["messages"][1].content == "Success"

    def test_decorator_response_rewriting(self) -> None:
        """Test decorator for rewriting responses."""

        @wrap_model_call
        def uppercase_responses(request, handler):
            result = handler(request.model_call)
            return ModelResponse(result=[AIMessage(content=result.result[0].content.upper())])

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
                return handler(request.model_call)
            except Exception:
                return ModelResponse(result=[AIMessage(content="Fallback response")])

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
            return handler(request.model_call)

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
            result = handler(request.model_call)
            execution_order.append("outer-after")
            return result

        @wrap_model_call
        def inner_middleware(request, handler):
            execution_order.append("inner-before")
            result = handler(request.model_call)
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
            return handler(request.model_call)

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
            return handler(request.model_call)

        assert isinstance(middleware_with_tools, AgentMiddleware)
        assert len(middleware_with_tools.tools) == 1
        assert middleware_with_tools.tools[0].name == "test_tool"

    def test_decorator_parentheses_optional(self) -> None:
        """Test that decorator works both with and without parentheses."""

        # Without parentheses
        @wrap_model_call
        def middleware_no_parens(request, handler):
            return handler(request.model_call)

        # With parentheses
        @wrap_model_call()
        def middleware_with_parens(request, handler):
            return handler(request.model_call)

        assert isinstance(middleware_no_parens, AgentMiddleware)
        assert isinstance(middleware_with_parens, AgentMiddleware)

    def test_decorator_preserves_function_name(self) -> None:
        """Test that decorator uses function name for class name."""

        @wrap_model_call
        def my_custom_middleware(request, handler):
            return handler(request.model_call)

        assert my_custom_middleware.__class__.__name__ == "my_custom_middleware"

    def test_decorator_mixed_with_class_middleware(self) -> None:
        """Test decorated middleware mixed with class-based middleware."""
        execution_order = []

        @wrap_model_call
        def decorated_middleware(request, handler):
            execution_order.append("decorated-before")
            result = handler(request.model_call)
            execution_order.append("decorated-after")
            return result

        class ClassMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                execution_order.append("class-before")
                result = handler(request.model_call)
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
            last_exception = None
            for attempt in range(max_retries):
                attempts.append(attempt + 1)
                try:
                    return handler(request.model_call)
                except Exception as e:
                    last_exception = e
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

    async def test_decorator_with_async_agent(self) -> None:
        """Test that decorated middleware works with async agent invocation."""
        call_log = []

        @wrap_model_call
        async def logging_middleware(request, handler):
            call_log.append("before")
            result = await handler(request.model_call)
            call_log.append("after")
            return result

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Async response")]))
        agent = create_agent(model=model, middleware=[logging_middleware])

        result = await agent.ainvoke({"messages": [HumanMessage("Test")]})

        assert call_log == ["before", "after"]
        assert result["messages"][1].content == "Async response"

    def test_decorator_request_modification(self) -> None:
        """Test decorator modifying request before execution."""
        modified_prompts = []

        @wrap_model_call
        def add_system_prompt(request, handler):
            # Modify request to add system prompt
            request.model_call.system_prompt = "You are a helpful assistant"
            modified_prompts.append(request.model_call.system_prompt)
            return handler(request.model_call)

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))
        agent = create_agent(model=model, middleware=[add_system_prompt])

        agent.invoke({"messages": [HumanMessage("Test")]})

        assert modified_prompts == ["You are a helpful assistant"]

    def test_decorator_multi_stage_transformation(self) -> None:
        """Test decorator applying multiple transformations."""

        @wrap_model_call
        def multi_transform(request, handler):
            result = handler(request.model_call)

            # First transformation: uppercase
            content = result.result[0].content.upper()
            # Second transformation: add prefix and suffix
            content = f"[START] {content} [END]"
            return ModelResponse(result=[AIMessage(content=content)])

        model = GenericFakeChatModel(messages=iter([AIMessage(content="hello")]))
        agent = create_agent(model=model, middleware=[multi_transform])

        result = agent.invoke({"messages": [HumanMessage("Test")]})

        assert result["messages"][1].content == "[START] HELLO [END]"
