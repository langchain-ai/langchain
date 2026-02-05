"""Unit tests for WrapModelCallResult state update support in wrap_model_call.

Tests that wrap_model_call middleware can return WrapModelCallResult to provide
state updates alongside the model response, with outermost middleware winning
on key conflicts.
"""

from collections.abc import Awaitable, Callable

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

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

    def test_state_update_overwrites_model_messages(self) -> None:
        """state_update with 'messages' key overwrites model response messages."""

        class OverwriteMessagesMiddleware(AgentMiddleware):
            def wrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], ModelResponse],
            ) -> WrapModelCallResult:
                response = handler(request)
                custom_msg = HumanMessage(content="Custom message", id="custom")
                return WrapModelCallResult(
                    model_response=response,
                    state_update={"messages": [custom_msg]},
                )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello!")]))
        agent = create_agent(model=model, middleware=[OverwriteMessagesMiddleware()])

        result = agent.invoke({"messages": [HumanMessage(content="Hi")]})

        # Model response messages are overwritten — only custom message survives
        messages = result["messages"]
        assert len(messages) == 2
        assert messages[0].content == "Hi"
        assert messages[1].content == "Custom message"

    def test_state_update_includes_model_messages_explicitly(self) -> None:
        """Middleware can include model messages alongside custom ones explicitly."""

        class ExplicitMessagesMiddleware(AgentMiddleware):
            def wrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], ModelResponse],
            ) -> WrapModelCallResult:
                response = handler(request)
                summary = HumanMessage(content="Summary", id="summary")
                return WrapModelCallResult(
                    model_response=response,
                    state_update={"messages": [summary, *response.result]},
                )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello!")]))
        agent = create_agent(model=model, middleware=[ExplicitMessagesMiddleware()])

        result = agent.invoke({"messages": [HumanMessage(content="Hi")]})

        messages = result["messages"]
        assert len(messages) == 3
        assert messages[0].content == "Hi"
        assert messages[1].content == "Summary"
        assert messages[2].content == "Hello!"

    def test_state_update_takes_priority_over_model_response(self) -> None:
        """state_update messages and structured_response take priority over model response."""

        class OverrideMiddleware(AgentMiddleware):
            def wrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], ModelResponse],
            ) -> WrapModelCallResult:
                response = handler(request)
                # Model response has its own messages and structured_response,
                # but state_update should win for both.
                response_with_structured = ModelResponse(
                    result=response.result,
                    structured_response={"from": "model"},
                )
                return WrapModelCallResult(
                    model_response=response_with_structured,
                    state_update={
                        "messages": [HumanMessage(content="From state_update", id="override")],
                        "structured_response": {"from": "state_update"},
                    },
                )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Model msg")]))
        agent = create_agent(model=model, middleware=[OverrideMiddleware()])

        result = agent.invoke({"messages": [HumanMessage("Hi")]})

        # state_update messages win over model response messages
        messages = result["messages"]
        assert len(messages) == 2
        assert messages[0].content == "Hi"
        assert messages[1].content == "From state_update"

        # state_update structured_response wins over model response structured_response
        assert result["structured_response"] == {"from": "state_update"}

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

    async def test_async_state_update_overwrites(self) -> None:
        """awrap_model_call state_update overwrites model response messages."""

        class AsyncOverwriteMiddleware(AgentMiddleware):
            async def awrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
            ) -> WrapModelCallResult:
                response = await handler(request)
                custom = HumanMessage(content="Async custom", id="async-custom")
                return WrapModelCallResult(
                    model_response=response,
                    state_update={"messages": [custom]},
                )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Async hello!")]))
        agent = create_agent(model=model, middleware=[AsyncOverwriteMiddleware()])

        result = await agent.ainvoke({"messages": [HumanMessage(content="Hi")]})

        messages = result["messages"]
        assert len(messages) == 2
        assert messages[0].content == "Hi"
        assert messages[1].content == "Async custom"

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
                state_update={
                    "messages": [
                        HumanMessage(content="Decorator msg", id="dec"),
                        *response.result,
                    ]
                },
            )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Async response")]))
        agent = create_agent(model=model, middleware=[state_update_middleware])

        result = await agent.ainvoke({"messages": [HumanMessage(content="Hi")]})

        messages = result["messages"]
        assert len(messages) == 3
        assert messages[1].content == "Decorator msg"
        assert messages[2].content == "Async response"


class TestComposition:
    """Test WrapModelCallResult with composed middleware.

    Key semantics: outermost middleware's state_update wins on key conflicts.
    """

    def test_outer_wrap_result_overwrites_model_messages(self) -> None:
        """Outer middleware's state_update overwrites model response messages."""
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
                    state_update={"messages": [HumanMessage(content="Outer msg", id="outer-msg")]},
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

        # Outer's state_update overwrites model messages
        messages = result["messages"]
        assert len(messages) == 2
        assert messages[0].content == "Hi"
        assert messages[1].content == "Outer msg"

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
                    state_update={
                        "messages": [
                            HumanMessage(content="Inner msg", id="inner"),
                            *response.result,
                        ]
                    },
                )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        agent = create_agent(
            model=model,
            middleware=[OuterMiddleware(), InnerMiddleware()],
        )

        result = agent.invoke({"messages": [HumanMessage("Hi")]})

        messages = result["messages"]
        assert len(messages) == 3
        assert messages[1].content == "Inner msg"
        assert messages[2].content == "Hello"

    def test_outer_state_update_wins_on_all_key_conflicts(self) -> None:
        """Outer's state_update fully overwrites inner's on all conflicting keys.

        This applies to all keys including 'messages' — no special casing.
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

        # Outer wins on all keys — inner's messages and custom_key are overwritten
        messages = result["messages"]
        assert len(messages) == 2
        assert messages[0].content == "Hi"
        assert messages[1].content == "Outer msg"

    def test_inner_state_preserved_when_outer_has_no_conflict(self) -> None:
        """Inner's state_update keys are preserved when outer doesn't conflict."""

        class MyState(AgentState):
            inner_key: str
            outer_key: str

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
                    state_update={"outer_key": "from_outer"},
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
                    state_update={"inner_key": "from_inner"},
                )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        agent = create_agent(
            model=model,
            middleware=[OuterMiddleware(), InnerMiddleware()],
        )

        result = agent.invoke({"messages": [HumanMessage("Hi")]})

        # Both keys survive since there's no conflict
        messages = result["messages"]
        assert messages[-1].content == "Hello"

    def test_inner_state_update_retry_safe(self) -> None:
        """When outer retries, only the last inner state update is used."""
        call_count = 0

        class MyState(AgentState):
            attempt: str

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
            state_schema = MyState  # type: ignore[assignment]

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
                    state_update={"attempt": f"attempt_{call_count}"},
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
        assert messages[-1].content == "Second"

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
                state_update={
                    "messages": [
                        HumanMessage(content="From decorator", id="dec"),
                        *response.result,
                    ]
                },
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
                response_with_structured = ModelResponse(
                    result=response.result,
                    structured_response={"key": "value"},
                )
                return WrapModelCallResult(
                    model_response=response_with_structured,
                    state_update={},
                )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        agent = create_agent(model=model, middleware=[StructuredMiddleware()])

        result = agent.invoke({"messages": [HumanMessage("Hi")]})

        assert result.get("structured_response") == {"key": "value"}
        messages = result["messages"]
        assert len(messages) == 2
        assert messages[1].content == "Hello"


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
                    state_update={
                        "messages": [
                            HumanMessage(content="Inner msg", id="inner"),
                            *response.result,
                        ]
                    },
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

    async def test_async_outer_wins_on_conflict(self) -> None:
        """Async: outer's state_update fully overwrites inner's on conflicts."""

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

        # Outer wins — inner's messages overwritten
        messages = result["messages"]
        assert len(messages) == 2
        assert messages[0].content == "Hi"
        assert messages[1].content == "Outer msg"

    async def test_async_inner_state_update_retry_safe(self) -> None:
        """Async: when outer retries, only last inner state update is used."""
        call_count = 0

        class MyState(AgentState):
            attempt: str

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
            state_schema = MyState  # type: ignore[assignment]

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
                    state_update={"attempt": f"attempt_{call_count}"},
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
        assert any(m.content == "Second" for m in messages)
