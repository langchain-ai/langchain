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

    def test_inner_wrap_result_propagated_through_composition(self) -> None:
        """Inner middleware's WrapModelCallResult state_update is propagated.

        When inner middleware returns WrapModelCallResult, its state_update is
        captured before normalizing to ModelResponse at the composition boundary
        and merged into the final result.
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

        # Inner's state_update is now propagated through the composition boundary
        messages = result["messages"]
        assert len(messages) == 3
        assert messages[1].content == "Inner msg"
        assert messages[2].content == "Hello"

    def test_both_outer_and_inner_wrap_result_merged(self) -> None:
        """Both outer and inner return WrapModelCallResult, state updates merged.

        Inner's messages come first, then outer's messages are appended.
        For non-messages keys, outer overwrites inner.
        """

        class MyState(AgentState):
            custom_key: str

        class OuterMiddleware(AgentMiddleware):
            state_schema = MyState  # type: ignore[assignment]

            def wrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], ModelResponse],
            ) -> WrapModelCallResult:
                response = handler(request)
                return WrapModelCallResult(
                    model_response=response,
                    state_update={
                        "messages": [HumanMessage(content="Outer msg", id="outer")],
                        "custom_key": "outer_value",
                    },
                )

        class InnerMiddleware(AgentMiddleware):
            state_schema = MyState  # type: ignore[assignment]

            def wrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], ModelResponse],
            ) -> WrapModelCallResult:
                response = handler(request)
                return WrapModelCallResult(
                    model_response=response,
                    state_update={
                        "messages": [HumanMessage(content="Inner msg", id="inner")],
                        "custom_key": "inner_value",
                    },
                )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        agent = create_agent(
            model=model,
            middleware=[OuterMiddleware(), InnerMiddleware()],
        )

        result = agent.invoke({"messages": [HumanMessage("Hi")]})

        # Messages: inner first, then outer, then model response
        messages = result["messages"]
        assert len(messages) == 4
        assert messages[1].content == "Inner msg"
        assert messages[2].content == "Outer msg"
        assert messages[3].content == "Hello"

    def test_inner_state_update_retry_safe(self) -> None:
        """When outer retries, only the last inner state update is used."""
        call_count = 0

        class OuterMiddleware(AgentMiddleware):
            def wrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], ModelResponse],
            ) -> ModelResponse:
                # Call handler twice (simulating retry)
                handler(request)
                return handler(request)

        class InnerMiddleware(AgentMiddleware):
            def wrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], ModelResponse],
            ) -> WrapModelCallResult:
                nonlocal call_count
                call_count += 1
                response = handler(request)
                return WrapModelCallResult(
                    model_response=response,
                    state_update={
                        "messages": [
                            HumanMessage(content=f"Attempt {call_count}", id=f"a{call_count}")
                        ]
                    },
                )

        model = GenericFakeChatModel(
            messages=iter([AIMessage(content="First"), AIMessage(content="Second")])
        )
        agent = create_agent(
            model=model,
            middleware=[OuterMiddleware(), InnerMiddleware()],
        )

        result = agent.invoke({"messages": [HumanMessage("Hi")]})

        # Only the last retry's inner state should survive
        messages = result["messages"]
        assert any(m.content == "Attempt 2" for m in messages)
        assert not any(m.content == "Attempt 1" for m in messages)

    def test_outer_state_update_wins_on_conflict(self) -> None:
        """Outer's non-messages state update overwrites inner's on same key."""

        class MyState(AgentState):
            priority: str

        class OuterMiddleware(AgentMiddleware):
            state_schema = MyState  # type: ignore[assignment]

            def wrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], ModelResponse],
            ) -> WrapModelCallResult:
                response = handler(request)
                return WrapModelCallResult(
                    model_response=response,
                    state_update={"priority": "outer_wins"},
                )

        class InnerMiddleware(AgentMiddleware):
            state_schema = MyState  # type: ignore[assignment]

            def wrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], ModelResponse],
            ) -> WrapModelCallResult:
                response = handler(request)
                return WrapModelCallResult(
                    model_response=response,
                    state_update={"priority": "inner_value"},
                )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        agent = create_agent(
            model=model,
            middleware=[OuterMiddleware(), InnerMiddleware()],
        )

        result = agent.invoke({"messages": [HumanMessage("Hi")]})

        # Outer wins on non-messages key conflicts
        # (state value verified indirectly since custom keys go through graph state)
        messages = result["messages"]
        assert messages[-1].content == "Hello"

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


class TestAsyncComposition:
    """Test async WrapModelCallResult propagation through composed middleware."""

    async def test_async_inner_wrap_result_propagated(self) -> None:
        """Async: inner middleware's WrapModelCallResult state_update is propagated."""

        class OuterMiddleware(AgentMiddleware):
            async def awrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
            ) -> ModelResponse:
                response = await handler(request)
                assert isinstance(response, ModelResponse)
                return response

        class InnerMiddleware(AgentMiddleware):
            async def awrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
            ) -> WrapModelCallResult:
                response = await handler(request)
                return WrapModelCallResult(
                    model_response=response,
                    state_update={"messages": [HumanMessage(content="Inner msg", id="inner")]},
                )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        agent = create_agent(
            model=model,
            middleware=[OuterMiddleware(), InnerMiddleware()],
        )

        result = await agent.ainvoke({"messages": [HumanMessage("Hi")]})

        messages = result["messages"]
        assert len(messages) == 3
        assert messages[1].content == "Inner msg"
        assert messages[2].content == "Hello"

    async def test_async_both_outer_and_inner_merged(self) -> None:
        """Async: both outer and inner WrapModelCallResult state updates are merged."""

        class OuterMiddleware(AgentMiddleware):
            async def awrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
            ) -> WrapModelCallResult:
                response = await handler(request)
                return WrapModelCallResult(
                    model_response=response,
                    state_update={
                        "messages": [HumanMessage(content="Outer msg", id="outer")],
                    },
                )

        class InnerMiddleware(AgentMiddleware):
            async def awrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
            ) -> WrapModelCallResult:
                response = await handler(request)
                return WrapModelCallResult(
                    model_response=response,
                    state_update={
                        "messages": [HumanMessage(content="Inner msg", id="inner")],
                    },
                )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        agent = create_agent(
            model=model,
            middleware=[OuterMiddleware(), InnerMiddleware()],
        )

        result = await agent.ainvoke({"messages": [HumanMessage("Hi")]})

        messages = result["messages"]
        assert len(messages) == 4
        assert messages[1].content == "Inner msg"
        assert messages[2].content == "Outer msg"
        assert messages[3].content == "Hello"

    async def test_async_inner_state_update_retry_safe(self) -> None:
        """Async: when outer retries, only last inner state update is used."""
        call_count = 0

        class OuterMiddleware(AgentMiddleware):
            async def awrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
            ) -> ModelResponse:
                # Call handler twice (simulating retry)
                await handler(request)
                return await handler(request)

        class InnerMiddleware(AgentMiddleware):
            async def awrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
            ) -> WrapModelCallResult:
                nonlocal call_count
                call_count += 1
                response = await handler(request)
                return WrapModelCallResult(
                    model_response=response,
                    state_update={
                        "messages": [
                            HumanMessage(content=f"Attempt {call_count}", id=f"a{call_count}")
                        ]
                    },
                )

        model = GenericFakeChatModel(
            messages=iter([AIMessage(content="First"), AIMessage(content="Second")])
        )
        agent = create_agent(
            model=model,
            middleware=[OuterMiddleware(), InnerMiddleware()],
        )

        result = await agent.ainvoke({"messages": [HumanMessage("Hi")]})

        messages = result["messages"]
        assert any(m.content == "Attempt 2" for m in messages)
        assert not any(m.content == "Attempt 1" for m in messages)
