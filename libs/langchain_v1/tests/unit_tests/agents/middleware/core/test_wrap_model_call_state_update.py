"""Unit tests for WrapModelCallResult command support in wrap_model_call.

Tests that wrap_model_call middleware can return WrapModelCallResult to provide
a Command alongside the model response. Commands are applied as separate state
updates through graph reducers (e.g. add_messages for messages).
"""

from collections.abc import Awaitable, Callable

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import InvalidUpdateError
from langgraph.types import Command

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    WrapModelCallResult,
    wrap_model_call,
)


class TestBasicCommand:
    """Test basic WrapModelCallResult functionality with Command."""

    def test_command_messages_added_alongside_model_messages(self) -> None:
        """Command messages are added alongside model response messages (additive)."""

        class AddMessagesMiddleware(AgentMiddleware):
            def wrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], ModelResponse],
            ) -> WrapModelCallResult:
                response = handler(request)
                custom_msg = HumanMessage(content="Custom message", id="custom")
                return WrapModelCallResult(
                    model_response=response,
                    command=Command(update={"messages": [custom_msg]}),
                )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello!")]))
        agent = create_agent(model=model, middleware=[AddMessagesMiddleware()])

        result = agent.invoke({"messages": [HumanMessage(content="Hi")]})

        # Both model response AND command messages appear (additive via add_messages)
        messages = result["messages"]
        assert len(messages) == 3
        assert messages[0].content == "Hi"
        assert messages[1].content == "Hello!"
        assert messages[2].content == "Custom message"

    def test_command_with_extra_messages_and_model_response(self) -> None:
        """Middleware can add extra messages via command alongside model messages."""

        class ExtraMessagesMiddleware(AgentMiddleware):
            def wrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], ModelResponse],
            ) -> WrapModelCallResult:
                response = handler(request)
                summary = HumanMessage(content="Summary", id="summary")
                return WrapModelCallResult(
                    model_response=response,
                    command=Command(update={"messages": [summary]}),
                )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello!")]))
        agent = create_agent(model=model, middleware=[ExtraMessagesMiddleware()])

        result = agent.invoke({"messages": [HumanMessage(content="Hi")]})

        messages = result["messages"]
        assert len(messages) == 3
        assert messages[0].content == "Hi"
        assert messages[1].content == "Hello!"
        assert messages[2].content == "Summary"

    def test_command_structured_response_conflicts_with_model_response(self) -> None:
        """Command and model response both setting structured_response raises."""

        class OverrideMiddleware(AgentMiddleware):
            def wrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], ModelResponse],
            ) -> WrapModelCallResult:
                response = handler(request)
                response_with_structured = ModelResponse(
                    result=response.result,
                    structured_response={"from": "model"},
                )
                return WrapModelCallResult(
                    model_response=response_with_structured,
                    command=Command(
                        update={
                            "structured_response": {"from": "command"},
                        }
                    ),
                )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Model msg")]))
        agent = create_agent(model=model, middleware=[OverrideMiddleware()])

        # Two Commands both setting structured_response (a LastValue channel)
        # in the same step raises InvalidUpdateError
        with pytest.raises(InvalidUpdateError):
            agent.invoke({"messages": [HumanMessage("Hi")]})

    def test_command_with_custom_state_field(self) -> None:
        """When command updates a custom field, model response messages are preserved."""

        class CustomFieldMiddleware(AgentMiddleware):
            def wrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], ModelResponse],
            ) -> WrapModelCallResult:
                response = handler(request)
                return WrapModelCallResult(
                    model_response=response,
                    command=Command(update={"custom_key": "custom_value"}),
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
                    command=Command(update={"summary": "conversation summarized"}),
                )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        agent = create_agent(model=model, middleware=[SummaryMiddleware()])

        result = agent.invoke({"messages": [HumanMessage("Hi")]})

        assert result["messages"][-1].content == "Hello"

    def test_no_command(self) -> None:
        """WrapModelCallResult with no command works like ModelResponse."""

        class NoCommandMiddleware(AgentMiddleware):
            def wrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], ModelResponse],
            ) -> WrapModelCallResult:
                response = handler(request)
                return WrapModelCallResult(
                    model_response=response,
                )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        agent = create_agent(model=model, middleware=[NoCommandMiddleware()])

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

    async def test_async_command_adds_messages(self) -> None:
        """awrap_model_call command adds messages alongside model response."""

        class AsyncAddMiddleware(AgentMiddleware):
            async def awrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
            ) -> WrapModelCallResult:
                response = await handler(request)
                custom = HumanMessage(content="Async custom", id="async-custom")
                return WrapModelCallResult(
                    model_response=response,
                    command=Command(update={"messages": [custom]}),
                )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Async hello!")]))
        agent = create_agent(model=model, middleware=[AsyncAddMiddleware()])

        result = await agent.ainvoke({"messages": [HumanMessage(content="Hi")]})

        # Both model response and command messages are present (additive)
        messages = result["messages"]
        assert len(messages) == 3
        assert messages[0].content == "Hi"
        assert messages[1].content == "Async hello!"
        assert messages[2].content == "Async custom"

    async def test_async_decorator_command(self) -> None:
        """@wrap_model_call async decorator returns WrapModelCallResult with command."""

        @wrap_model_call
        async def command_middleware(
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        ) -> WrapModelCallResult:
            response = await handler(request)
            return WrapModelCallResult(
                model_response=response,
                command=Command(
                    update={
                        "messages": [
                            HumanMessage(content="Decorator msg", id="dec"),
                        ]
                    }
                ),
            )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Async response")]))
        agent = create_agent(model=model, middleware=[command_middleware])

        result = await agent.ainvoke({"messages": [HumanMessage(content="Hi")]})

        messages = result["messages"]
        assert len(messages) == 3
        assert messages[1].content == "Async response"
        assert messages[2].content == "Decorator msg"


class TestComposition:
    """Test WrapModelCallResult with composed middleware.

    Key semantics: Commands are collected inner-first, then outer.
    For non-reducer fields, later Commands overwrite (outer wins).
    For reducer fields (messages), all Commands are additive.
    """

    def test_outer_command_messages_added_alongside_model(self) -> None:
        """Outer middleware's command messages are added alongside model messages."""
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
                    command=Command(
                        update={"messages": [HumanMessage(content="Outer msg", id="outer-msg")]}
                    ),
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

        # Model messages + outer command messages (additive)
        messages = result["messages"]
        assert len(messages) == 3
        assert messages[0].content == "Hi"
        assert messages[1].content == "Composed"
        assert messages[2].content == "Outer msg"

    def test_inner_command_propagated_through_composition(self) -> None:
        """Inner middleware's WrapModelCallResult command is propagated.

        When inner middleware returns WrapModelCallResult, its command is
        captured before normalizing to ModelResponse at the composition boundary
        and collected into the final result.
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
                    command=Command(
                        update={
                            "messages": [
                                HumanMessage(content="Inner msg", id="inner"),
                            ]
                        }
                    ),
                )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        agent = create_agent(
            model=model,
            middleware=[OuterMiddleware(), InnerMiddleware()],
        )

        result = agent.invoke({"messages": [HumanMessage("Hi")]})

        # Model messages + inner command messages (additive)
        messages = result["messages"]
        assert len(messages) == 3
        assert messages[0].content == "Hi"
        assert messages[1].content == "Hello"
        assert messages[2].content == "Inner msg"

    def test_non_reducer_key_conflict_raises(self) -> None:
        """Multiple Commands setting the same non-reducer key raises.

        LastValue channels (like custom_key) can only receive one value per
        step. Inner and outer both setting the same key is an error.
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
                    command=Command(
                        update={
                            "messages": [HumanMessage(content="Outer msg", id="outer")],
                            "custom_key": "outer_value",
                        }
                    ),
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
                    command=Command(
                        update={
                            "messages": [HumanMessage(content="Inner msg", id="inner")],
                            "custom_key": "inner_value",
                        }
                    ),
                )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        agent = create_agent(
            model=model,
            middleware=[OuterMiddleware(), InnerMiddleware()],
        )

        # Two Commands both setting custom_key (a LastValue channel)
        # in the same step raises InvalidUpdateError
        with pytest.raises(InvalidUpdateError):
            agent.invoke({"messages": [HumanMessage("Hi")]})

    def test_inner_state_preserved_when_outer_has_no_conflict(self) -> None:
        """Inner's command keys are preserved when outer doesn't conflict."""

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
                    command=Command(update={"outer_key": "from_outer"}),
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
                    command=Command(update={"inner_key": "from_inner"}),
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

    def test_inner_command_retry_safe(self) -> None:
        """When outer retries, only the last inner command is used."""
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
                    command=Command(update={"attempt": f"attempt_{call_count}"}),
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
        """@wrap_model_call decorator can return WrapModelCallResult with command."""

        @wrap_model_call
        def command_middleware(
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> WrapModelCallResult:
            response = handler(request)
            return WrapModelCallResult(
                model_response=response,
                command=Command(
                    update={
                        "messages": [
                            HumanMessage(content="From decorator", id="dec"),
                        ]
                    }
                ),
            )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Model response")]))
        agent = create_agent(model=model, middleware=[command_middleware])

        result = agent.invoke({"messages": [HumanMessage("Hi")]})

        messages = result["messages"]
        assert len(messages) == 3
        assert messages[1].content == "Model response"
        assert messages[2].content == "From decorator"

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

    async def test_async_inner_command_propagated(self) -> None:
        """Async: inner middleware's WrapModelCallResult command is propagated."""

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
                    command=Command(
                        update={
                            "messages": [
                                HumanMessage(content="Inner msg", id="inner"),
                            ]
                        }
                    ),
                )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        agent = create_agent(
            model=model,
            middleware=[OuterMiddleware(), InnerMiddleware()],
        )

        result = await agent.ainvoke({"messages": [HumanMessage("Hi")]})

        # Model messages + inner command messages (additive)
        messages = result["messages"]
        assert len(messages) == 3
        assert messages[0].content == "Hi"
        assert messages[1].content == "Hello"
        assert messages[2].content == "Inner msg"

    async def test_async_both_commands_additive_messages(self) -> None:
        """Async: both inner and outer command messages are added alongside model."""

        class OuterMiddleware(AgentMiddleware):
            async def awrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
            ) -> WrapModelCallResult:
                response = await handler(request)
                return WrapModelCallResult(
                    model_response=response,
                    command=Command(
                        update={"messages": [HumanMessage(content="Outer msg", id="outer")]}
                    ),
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
                    command=Command(
                        update={"messages": [HumanMessage(content="Inner msg", id="inner")]}
                    ),
                )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        agent = create_agent(
            model=model,
            middleware=[OuterMiddleware(), InnerMiddleware()],
        )

        result = await agent.ainvoke({"messages": [HumanMessage("Hi")]})

        # All messages additive: model + inner + outer
        messages = result["messages"]
        assert len(messages) == 4
        assert messages[0].content == "Hi"
        assert messages[1].content == "Hello"
        assert messages[2].content == "Inner msg"
        assert messages[3].content == "Outer msg"

    async def test_async_inner_command_retry_safe(self) -> None:
        """Async: when outer retries, only last inner command is used."""
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
                    command=Command(update={"attempt": f"attempt_{call_count}"}),
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


class TestCommandGotoDisallowed:
    """Test that Command goto raises NotImplementedError in wrap_model_call."""

    def test_command_goto_raises_not_implemented(self) -> None:
        """Command with goto in wrap_model_call raises NotImplementedError."""

        class GotoMiddleware(AgentMiddleware):
            def wrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], ModelResponse],
            ) -> WrapModelCallResult:
                response = handler(request)
                return WrapModelCallResult(
                    model_response=response,
                    command=Command(goto="__end__"),
                )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello!")]))
        agent = create_agent(model=model, middleware=[GotoMiddleware()])

        with pytest.raises(NotImplementedError, match="Command goto is not yet supported"):
            agent.invoke({"messages": [HumanMessage(content="Hi")]})

    async def test_async_command_goto_raises_not_implemented(self) -> None:
        """Async: Command with goto in wrap_model_call raises NotImplementedError."""

        class AsyncGotoMiddleware(AgentMiddleware):
            async def awrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
            ) -> WrapModelCallResult:
                response = await handler(request)
                return WrapModelCallResult(
                    model_response=response,
                    command=Command(goto="tools"),
                )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello!")]))
        agent = create_agent(model=model, middleware=[AsyncGotoMiddleware()])

        with pytest.raises(NotImplementedError, match="Command goto is not yet supported"):
            await agent.ainvoke({"messages": [HumanMessage(content="Hi")]})


class TestCommandResumeDisallowed:
    """Test that Command resume raises NotImplementedError in wrap_model_call."""

    def test_command_resume_raises_not_implemented(self) -> None:
        """Command with resume in wrap_model_call raises NotImplementedError."""

        class ResumeMiddleware(AgentMiddleware):
            def wrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], ModelResponse],
            ) -> WrapModelCallResult:
                response = handler(request)
                return WrapModelCallResult(
                    model_response=response,
                    command=Command(resume="some_value"),
                )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello!")]))
        agent = create_agent(model=model, middleware=[ResumeMiddleware()])

        with pytest.raises(NotImplementedError, match="Command resume is not yet supported"):
            agent.invoke({"messages": [HumanMessage(content="Hi")]})

    async def test_async_command_resume_raises_not_implemented(self) -> None:
        """Async: Command with resume in wrap_model_call raises NotImplementedError."""

        class AsyncResumeMiddleware(AgentMiddleware):
            async def awrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
            ) -> WrapModelCallResult:
                response = await handler(request)
                return WrapModelCallResult(
                    model_response=response,
                    command=Command(resume="some_value"),
                )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello!")]))
        agent = create_agent(model=model, middleware=[AsyncResumeMiddleware()])

        with pytest.raises(NotImplementedError, match="Command resume is not yet supported"):
            await agent.ainvoke({"messages": [HumanMessage(content="Hi")]})


class TestCommandGraphDisallowed:
    """Test that Command graph raises NotImplementedError in wrap_model_call."""

    def test_command_graph_raises_not_implemented(self) -> None:
        """Command with graph in wrap_model_call raises NotImplementedError."""

        class GraphMiddleware(AgentMiddleware):
            def wrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], ModelResponse],
            ) -> WrapModelCallResult:
                response = handler(request)
                return WrapModelCallResult(
                    model_response=response,
                    command=Command(graph=Command.PARENT, update={"messages": []}),
                )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello!")]))
        agent = create_agent(model=model, middleware=[GraphMiddleware()])

        with pytest.raises(NotImplementedError, match="Command graph is not yet supported"):
            agent.invoke({"messages": [HumanMessage(content="Hi")]})

    async def test_async_command_graph_raises_not_implemented(self) -> None:
        """Async: Command with graph in wrap_model_call raises NotImplementedError."""

        class AsyncGraphMiddleware(AgentMiddleware):
            async def awrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
            ) -> WrapModelCallResult:
                response = await handler(request)
                return WrapModelCallResult(
                    model_response=response,
                    command=Command(graph=Command.PARENT, update={"messages": []}),
                )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello!")]))
        agent = create_agent(model=model, middleware=[AsyncGraphMiddleware()])

        with pytest.raises(NotImplementedError, match="Command graph is not yet supported"):
            await agent.ainvoke({"messages": [HumanMessage(content="Hi")]})
