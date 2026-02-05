"""Unit tests for WrapModelCallResult state update support in wrap_model_call.

Tests that wrap_model_call middleware can return WrapModelCallResult to provide
state updates alongside the model response.
"""

from collections.abc import Awaitable, Callable

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    WrapModelCallResult,
    wrap_model_call,
)


class TestBasicStateUpdate:
    """Test basic WrapModelCallResult functionality."""

    def test_state_update_with_messages(self) -> None:
        """Middleware returns WrapModelCallResult with extra messages in state_update.

        The state_update messages should be prepended before the model response messages.
        """

        class PrependMessageMiddleware(AgentMiddleware):
            def wrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], ModelResponse],
            ) -> WrapModelCallResult:
                response = handler(request)
                # Add a RemoveMessage + summary before the AI response
                summary = HumanMessage(content="Summary of prior conversation", id="summary")
                return WrapModelCallResult(
                    model_response=response,
                    state_update={
                        "messages": [summary],
                    },
                )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello!")]))
        agent = create_agent(model=model, middleware=[PrependMessageMiddleware()])

        result = agent.invoke({"messages": [HumanMessage(content="Hi")]})

        # Should have: original HumanMessage, summary message, AI response
        messages = result["messages"]
        assert len(messages) == 3
        # The summary is prepended before model response via add_messages reducer
        assert messages[1].content == "Summary of prior conversation"
        assert messages[2].content == "Hello!"

    def test_state_update_with_remove_and_new_messages(self) -> None:
        """Middleware uses RemoveMessage in state_update to clear history."""

        class ClearHistoryMiddleware(AgentMiddleware):
            def wrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], ModelResponse],
            ) -> WrapModelCallResult:
                response = handler(request)
                # Remove the original human message, add a summary before the AI response
                remove_ops = [RemoveMessage(id=m.id) for m in request.state["messages"] if m.id]
                return WrapModelCallResult(
                    model_response=response,
                    state_update={
                        "messages": remove_ops,
                    },
                )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))
        agent = create_agent(model=model, middleware=[ClearHistoryMiddleware()])

        result = agent.invoke({"messages": [HumanMessage(content="Hi", id="msg1")]})

        # After RemoveMessage(id="msg1"), then AI response added
        messages = result["messages"]
        # The original message was removed, only AI response remains
        assert len(messages) == 1
        assert messages[0].content == "Response"

    def test_state_update_without_messages_key(self) -> None:
        """When state_update doesn't include 'messages', model response messages are used."""

        class CustomFieldMiddleware(AgentMiddleware):
            def wrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], ModelResponse],
            ) -> WrapModelCallResult:
                response = handler(request)
                return WrapModelCallResult(
                    model_response=response,
                    state_update={"custom_key": "custom_value"},
                )

        class CustomState(AgentState):
            custom_key: str

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        agent = create_agent(
            model=model,
            middleware=[CustomFieldMiddleware()],
            state_schema=CustomState,
        )

        result = agent.invoke({"messages": [HumanMessage("Hi")]})

        assert result["messages"][-1].content == "Hello"


class TestCustomStateField:
    """Test WrapModelCallResult with custom state fields defined via state_schema."""

    def test_custom_field_via_state_schema(self) -> None:
        """Middleware updates a custom state field via WrapModelCallResult."""

        class MyState(AgentState):
            summary: str

        class SummaryMiddleware(AgentMiddleware):
            state_schema = MyState  # type: ignore[assignment]

            def wrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], ModelResponse],
            ) -> WrapModelCallResult:
                response = handler(request)
                return WrapModelCallResult(
                    model_response=response,
                    state_update={"summary": "conversation summarized"},
                )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        agent = create_agent(model=model, middleware=[SummaryMiddleware()])

        result = agent.invoke({"messages": [HumanMessage("Hi")]})

        assert result["messages"][-1].content == "Hello"

    def test_empty_state_update(self) -> None:
        """WrapModelCallResult with empty state_update works like ModelResponse."""

        class EmptyUpdateMiddleware(AgentMiddleware):
            def wrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], ModelResponse],
            ) -> WrapModelCallResult:
                response = handler(request)
                return WrapModelCallResult(
                    model_response=response,
                    state_update={},
                )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        agent = create_agent(model=model, middleware=[EmptyUpdateMiddleware()])

        result = agent.invoke({"messages": [HumanMessage("Hi")]})

        assert len(result["messages"]) == 2
        assert result["messages"][1].content == "Hello"


class TestBackwardsCompatibility:
    """Test that existing ModelResponse and AIMessage returns still work."""

    def test_model_response_return_unchanged(self) -> None:
        """Existing middleware returning ModelResponse works identically."""

        class PassthroughMiddleware(AgentMiddleware):
            def wrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], ModelResponse],
            ) -> ModelResponse:
                return handler(request)

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        agent = create_agent(model=model, middleware=[PassthroughMiddleware()])

        result = agent.invoke({"messages": [HumanMessage("Hi")]})

        assert len(result["messages"]) == 2
        assert result["messages"][1].content == "Hello"

    def test_ai_message_return_unchanged(self) -> None:
        """Existing middleware returning AIMessage works identically."""

        class ShortCircuitMiddleware(AgentMiddleware):
            def wrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], ModelResponse],
            ) -> AIMessage:
                return AIMessage(content="Short-circuited")

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Should not appear")]))
        agent = create_agent(model=model, middleware=[ShortCircuitMiddleware()])

        result = agent.invoke({"messages": [HumanMessage("Hi")]})

        assert len(result["messages"]) == 2
        assert result["messages"][1].content == "Short-circuited"

    def test_no_middleware_unchanged(self) -> None:
        """Agent without middleware works identically."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        agent = create_agent(model=model)

        result = agent.invoke({"messages": [HumanMessage("Hi")]})

        assert len(result["messages"]) == 2
        assert result["messages"][1].content == "Hello"


class TestAsyncWrapModelCallResult:
    """Test async variant of WrapModelCallResult."""

    async def test_async_state_update(self) -> None:
        """awrap_model_call returns WrapModelCallResult with state updates."""

        class AsyncStateUpdateMiddleware(AgentMiddleware):
            async def awrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
            ) -> WrapModelCallResult:
                response = await handler(request)
                summary = HumanMessage(content="Async summary", id="async-summary")
                return WrapModelCallResult(
                    model_response=response,
                    state_update={"messages": [summary]},
                )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Async hello!")]))
        agent = create_agent(model=model, middleware=[AsyncStateUpdateMiddleware()])

        result = await agent.ainvoke({"messages": [HumanMessage(content="Hi")]})

        messages = result["messages"]
        assert len(messages) == 3
        assert messages[1].content == "Async summary"
        assert messages[2].content == "Async hello!"

    async def test_async_decorator_state_update(self) -> None:
        """@wrap_model_call async decorator returns WrapModelCallResult."""

        @wrap_model_call
        async def state_update_middleware(
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        ) -> WrapModelCallResult:
            response = await handler(request)
            return WrapModelCallResult(
                model_response=response,
                state_update={"messages": [HumanMessage(content="Decorator summary", id="dec")]},
            )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Async response")]))
        agent = create_agent(model=model, middleware=[state_update_middleware])

        result = await agent.ainvoke({"messages": [HumanMessage(content="Hi")]})

        messages = result["messages"]
        assert len(messages) == 3
        assert messages[1].content == "Decorator summary"
        assert messages[2].content == "Async response"


class TestComposition:
    """Test WrapModelCallResult with composed middleware."""

    def test_outer_wrap_result_inner_model_response(self) -> None:
        """Outer middleware returns WrapModelCallResult, inner returns ModelResponse.

        The outer's state_update should be applied and inner should work normally.
        """
        execution_order: list[str] = []

        class OuterMiddleware(AgentMiddleware):
            def wrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], ModelResponse],
            ) -> WrapModelCallResult:
                execution_order.append("outer-before")
                response = handler(request)
                execution_order.append("outer-after")
                return WrapModelCallResult(
                    model_response=response,
                    state_update={
                        "messages": [HumanMessage(content="Outer state msg", id="outer-msg")]
                    },
                )

        class InnerMiddleware(AgentMiddleware):
            def wrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], ModelResponse],
            ) -> ModelResponse:
                execution_order.append("inner-before")
                response = handler(request)
                execution_order.append("inner-after")
                return response

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Composed")]))
        agent = create_agent(
            model=model,
            middleware=[OuterMiddleware(), InnerMiddleware()],
        )

        result = agent.invoke({"messages": [HumanMessage("Hi")]})

        # Execution order: outer wraps inner
        assert execution_order == [
            "outer-before",
            "inner-before",
            "inner-after",
            "outer-after",
        ]

        # Outer's state_update messages prepend before model response
        messages = result["messages"]
        assert len(messages) == 3
        assert messages[1].content == "Outer state msg"
        assert messages[2].content == "Composed"

    def test_inner_wrap_result_dropped_at_composition_boundary(self) -> None:
        """Inner middleware's WrapModelCallResult is normalized away for outer.

        When inner middleware returns WrapModelCallResult, it gets normalized
        to ModelResponse at the composition boundary. Only the outermost
        WrapModelCallResult is preserved.
        """

        class OuterMiddleware(AgentMiddleware):
            def wrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], ModelResponse],
            ) -> ModelResponse:
                # Outer sees a ModelResponse from handler (inner's WrapModelCallResult
                # was normalized at the composition boundary)
                response = handler(request)
                assert isinstance(response, ModelResponse)
                return response

        class InnerMiddleware(AgentMiddleware):
            def wrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], ModelResponse],
            ) -> WrapModelCallResult:
                response = handler(request)
                return WrapModelCallResult(
                    model_response=response,
                    state_update={"messages": [HumanMessage(content="Inner msg", id="inner")]},
                )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        agent = create_agent(
            model=model,
            middleware=[OuterMiddleware(), InnerMiddleware()],
        )

        result = agent.invoke({"messages": [HumanMessage("Hi")]})

        # Inner's state_update was dropped because outer returned ModelResponse
        messages = result["messages"]
        assert len(messages) == 2
        assert messages[1].content == "Hello"

    def test_decorator_returns_wrap_result(self) -> None:
        """@wrap_model_call decorator can return WrapModelCallResult."""

        @wrap_model_call
        def state_update_middleware(
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> WrapModelCallResult:
            response = handler(request)
            return WrapModelCallResult(
                model_response=response,
                state_update={"messages": [HumanMessage(content="From decorator", id="dec")]},
            )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Model response")]))
        agent = create_agent(model=model, middleware=[state_update_middleware])

        result = agent.invoke({"messages": [HumanMessage("Hi")]})

        messages = result["messages"]
        assert len(messages) == 3
        assert messages[1].content == "From decorator"
        assert messages[2].content == "Model response"

    def test_structured_response_preserved(self) -> None:
        """WrapModelCallResult preserves structured_response from ModelResponse."""

        class StructuredMiddleware(AgentMiddleware):
            def wrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], ModelResponse],
            ) -> WrapModelCallResult:
                response = handler(request)
                # Simulate a structured response being set
                response_with_structured = ModelResponse(
                    result=response.result,
                    structured_response={"key": "value"},
                )
                return WrapModelCallResult(
                    model_response=response_with_structured,
                    state_update={"messages": [HumanMessage(content="Extra", id="extra")]},
                )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        agent = create_agent(model=model, middleware=[StructuredMiddleware()])

        result = agent.invoke({"messages": [HumanMessage("Hi")]})

        assert result.get("structured_response") == {"key": "value"}
        messages = result["messages"]
        assert len(messages) == 3
        assert messages[1].content == "Extra"
        assert messages[2].content == "Hello"
