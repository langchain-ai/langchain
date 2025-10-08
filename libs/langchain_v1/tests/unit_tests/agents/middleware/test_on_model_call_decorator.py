"""Unit tests for the @on_model_call decorator."""

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from langchain.agents import create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    on_model_call,
)


class TestOnModelCallDecorator:
    """Test the @on_model_call decorator for creating middleware."""

    def test_basic_decorator_usage(self) -> None:
        """Test basic decorator usage without parameters."""

        @on_model_call
        def passthrough_middleware(request, state, runtime):
            response = yield request
            # Generator ends

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

        @on_model_call(name="CustomMiddleware")
        def my_middleware(request, state, runtime):
            response = yield request
            # Generator ends

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

        @on_model_call
        def retry_once(request, state, runtime):
            try:
                result = yield request
                # Generator ends
            except Exception:
                # Retry once
                result = yield request
                # Generator ends

        model = FailOnceThenSucceed(messages=iter([AIMessage(content="Success")]))
        agent = create_agent(model=model, middleware=[retry_once])

        result = agent.invoke({"messages": [HumanMessage("Test")]})

        assert call_count["value"] == 2
        assert result["messages"][1].content == "Success"

    def test_decorator_response_rewriting(self) -> None:
        """Test decorator for rewriting responses."""

        @on_model_call
        def uppercase_responses(request, state, runtime):
            result = yield request
            modified = AIMessage(content=result.content.upper())
            yield modified

        model = GenericFakeChatModel(messages=iter([AIMessage(content="hello world")]))
        agent = create_agent(model=model, middleware=[uppercase_responses])

        result = agent.invoke({"messages": [HumanMessage("Test")]})

        assert result["messages"][1].content == "HELLO WORLD"

    def test_decorator_error_handling(self) -> None:
        """Test decorator for error recovery."""

        class AlwaysFailModel(GenericFakeChatModel):
            def _generate(self, messages, **kwargs):
                raise ValueError("Model error")

        @on_model_call
        def error_to_fallback(request, state, runtime):
            try:
                result = yield request
            except Exception:
                fallback = AIMessage(content="Fallback response")
                yield fallback

        model = AlwaysFailModel(messages=iter([]))
        agent = create_agent(model=model, middleware=[error_to_fallback])

        result = agent.invoke({"messages": [HumanMessage("Test")]})

        assert result["messages"][1].content == "Fallback response"

    def test_decorator_with_state_access(self) -> None:
        """Test decorator accessing agent state."""
        state_values = []

        @on_model_call
        def log_state(request, state, runtime):
            state_values.append(state.get("messages"))
            response = yield request
            # Generator ends

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

        @on_model_call
        def outer_middleware(request, state, runtime):
            execution_order.append("outer-before")
            response = yield request
            execution_order.append("outer-after")
            # Generator ends

        @on_model_call
        def inner_middleware(request, state, runtime):
            execution_order.append("inner-before")
            response = yield request
            execution_order.append("inner-after")
            # Generator ends

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

        @on_model_call(state_schema=CustomState)
        def middleware_with_schema(request, state, runtime):
            response = yield request
            # Generator ends

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

        @on_model_call(tools=[test_tool])
        def middleware_with_tools(request, state, runtime):
            response = yield request
            # Generator ends

        assert isinstance(middleware_with_tools, AgentMiddleware)
        assert len(middleware_with_tools.tools) == 1
        assert middleware_with_tools.tools[0].name == "test_tool"

    def test_decorator_parentheses_optional(self) -> None:
        """Test that decorator works both with and without parentheses."""

        # Without parentheses
        @on_model_call
        def middleware_no_parens(request, state, runtime):
            response = yield request
            # Generator ends

        # With parentheses
        @on_model_call()
        def middleware_with_parens(request, state, runtime):
            response = yield request
            # Generator ends

        assert isinstance(middleware_no_parens, AgentMiddleware)
        assert isinstance(middleware_with_parens, AgentMiddleware)

    def test_decorator_preserves_function_name(self) -> None:
        """Test that decorator uses function name for class name."""

        @on_model_call
        def my_custom_middleware(request, state, runtime):
            response = yield request
            # Generator ends

        assert my_custom_middleware.__class__.__name__ == "my_custom_middleware"

    def test_decorator_mixed_with_class_middleware(self) -> None:
        """Test decorated middleware mixed with class-based middleware."""
        execution_order = []

        @on_model_call
        def decorated_middleware(request, state, runtime):
            execution_order.append("decorated-before")
            response = yield request
            execution_order.append("decorated-after")
            # Generator ends

        class ClassMiddleware(AgentMiddleware):
            def on_model_call(self, request, state, runtime):
                execution_order.append("class-before")
                response = yield request
                execution_order.append("class-after")
                # Generator ends

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

        @on_model_call
        def retry_with_tracking(request, state, runtime):
            max_retries = 3
            for attempt in range(max_retries):
                attempts.append(attempt + 1)
                try:
                    result = yield request
                    # Generator ends
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

    async def test_decorator_with_async_agent(self) -> None:
        """Test that decorated middleware works with async agent invocation."""
        call_log = []

        @on_model_call
        def logging_middleware(request, state, runtime):
            call_log.append("before")
            response = yield request
            call_log.append("after")
            # Generator ends

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Async response")]))
        agent = create_agent(model=model, middleware=[logging_middleware])

        result = await agent.ainvoke({"messages": [HumanMessage("Test")]})

        assert call_log == ["before", "after"]
        assert result["messages"][1].content == "Async response"

    def test_decorator_request_modification(self) -> None:
        """Test decorator modifying request before execution."""
        modified_prompts = []

        @on_model_call
        def add_system_prompt(request, state, runtime):
            # Modify request to add system prompt
            modified_request = ModelRequest(
                messages=request.messages,
                model=request.model,
                system_prompt="You are a helpful assistant",
                tool_choice=request.tool_choice,
                tools=request.tools,
                response_format=request.response_format,
            )
            modified_prompts.append(modified_request.system_prompt)
            response = yield modified_request
            # Generator ends

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))
        agent = create_agent(model=model, middleware=[add_system_prompt])

        agent.invoke({"messages": [HumanMessage("Test")]})

        assert modified_prompts == ["You are a helpful assistant"]

    def test_decorator_multi_stage_transformation(self) -> None:
        """Test decorator applying multiple transformations."""

        @on_model_call
        def multi_transform(request, state, runtime):
            result = yield request

            # First transformation: uppercase
            content = result.content.upper()
            # Second transformation: add prefix and suffix
            content = f"[START] {content} [END]"
            modified = AIMessage(content=content)
            yield modified

        model = GenericFakeChatModel(messages=iter([AIMessage(content="hello")]))
        agent = create_agent(model=model, middleware=[multi_transform])

        result = agent.invoke({"messages": [HumanMessage("Test")]})

        assert result["messages"][1].content == "[START] HELLO [END]"
