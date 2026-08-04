"""Unit tests for LLM tool selection middleware."""

from collections.abc import Callable, Iterator, Sequence
from itertools import cycle
from typing import Any, Literal

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel
from typing_extensions import override

from langchain.agents import create_agent
from langchain.agents.middleware import (
    LLMToolSelectorMiddleware,
    ModelRequest,
    ModelResponse,
    wrap_model_call,
)
from langchain.agents.middleware.tool_selection import _create_tool_selection_response
from langchain.messages import AIMessage


@tool
def get_weather(location: str) -> str:
    """Get current weather for a location."""
    return f"Weather in {location}: 72°F, sunny"


@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"


@tool
def calculate(expression: str) -> str:
    """Perform mathematical calculations."""
    return f"Result of {expression}: 42"


@tool
def send_email(to: str, subject: str) -> str:
    """Send an email to someone."""
    return f"Email with subject {subject} sent to {to}"


@tool
def get_stock_price(symbol: str) -> str:
    """Get current stock price for a symbol."""
    return f"Stock price for {symbol}: $150.25"


class FakeModel(GenericFakeChatModel):
    tool_style: Literal["openai", "anthropic"] = "openai"

    @override
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type[BaseModel] | Callable[..., Any] | BaseTool],
        **_kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        if len(tools) == 0:
            msg = "Must provide at least one tool"
            raise ValueError(msg)

        tool_dicts = []
        for tool_ in tools:
            if isinstance(tool_, dict):
                tool_dicts.append(tool_)
                continue
            if not isinstance(tool_, BaseTool):
                msg = "Only BaseTool and dict is supported by FakeToolCallingModel.bind_tools"
                raise TypeError(msg)

            # NOTE: this is a simplified tool spec for testing purposes only
            if self.tool_style == "openai":
                tool_dicts.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool_.name,
                        },
                    }
                )
            elif self.tool_style == "anthropic":
                tool_dicts.append(
                    {
                        "name": tool_.name,
                    }
                )

        return self.bind(tools=tool_dicts)


class _StreamLeakDetectingModel(FakeModel):
    """Detects whether the streaming code path is used for this model call.

    `BaseChatModel` only routes an `invoke()` call through `_stream()` (instead of
    `_generate()`) when a streaming callback handler is attached -- exactly the
    condition isolation is meant to prevent for middleware-internal calls. Real
    providers would additionally leak partial tool-call JSON in that scenario;
    `GenericFakeChatModel` doesn't chunk the `tool_calls` field, so this override
    tracks whether `_stream` ran at all, which is the precise mechanism under test.
    """

    stream_was_called: bool = False

    @override
    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        self.stream_was_called = True
        yield from super()._stream(*args, **kwargs)


class _ChatModelStartCounter(BaseCallbackHandler):
    """Counts `on_chat_model_start` invocations."""

    def __init__(self) -> None:
        self.count = 0

    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        self.count += 1


class _StreamingChatModelStartCounter(_ChatModelStartCounter):
    """Same as `_ChatModelStartCounter`, but structurally a streaming handler.

    Implementing `tap_output_aiter`/`tap_output_iter` satisfies the
    `runtime_checkable` `_StreamingCallbackHandler` protocol, matching how
    `stream_mode="messages"`/`astream_events`/`astream_log` handlers are identified.
    """

    def tap_output_aiter(self, run_id: Any, output: Any) -> Any:  # noqa: ARG002
        return output

    def tap_output_iter(self, run_id: Any, output: Any) -> Any:  # noqa: ARG002
        return output


class TestCallbackIsolation:
    """Test that the internal selection call doesn't leak into the parent's stream.

    The internal call should still be traced/observed by non-streaming callbacks
    (e.g. LangSmith), but must not emit to handlers that would surface its tokens
    in the user-facing stream.
    """

    def test_sync_streaming_handler_only_sees_main_call(self) -> None:
        """A streaming-marker handler should only observe the main model call."""
        tool_selection_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "ToolSelectionResponse",
                                "id": "1",
                                "args": {"tools": ["get_weather"]},
                            }
                        ],
                    ),
                ]
            )
        )
        model = FakeModel(messages=iter([AIMessage(content="Done")]))
        tool_selector = LLMToolSelectorMiddleware(max_tools=1, model=tool_selection_model)

        agent = create_agent(
            model=model,
            tools=[get_weather, search_web],
            middleware=[tool_selector],
        )

        streaming_counter = _StreamingChatModelStartCounter()
        non_streaming_counter = _ChatModelStartCounter()
        agent.invoke(
            {"messages": [HumanMessage("weather in paris")]},
            config={"callbacks": [streaming_counter, non_streaming_counter]},
        )

        # Streaming handler: only the main model call, not the internal selector call.
        assert streaming_counter.count == 1
        # Plain (e.g. tracer-like) handler: both calls remain observable.
        assert non_streaming_counter.count == 2

    async def test_async_streaming_handler_only_sees_main_call(self) -> None:
        """A streaming-marker handler should only observe the main model call."""
        tool_selection_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "ToolSelectionResponse",
                                "id": "1",
                                "args": {"tools": ["get_weather"]},
                            }
                        ],
                    ),
                ]
            )
        )
        model = FakeModel(messages=iter([AIMessage(content="Done")]))
        tool_selector = LLMToolSelectorMiddleware(max_tools=1, model=tool_selection_model)

        agent = create_agent(
            model=model,
            tools=[get_weather, search_web],
            middleware=[tool_selector],
        )

        streaming_counter = _StreamingChatModelStartCounter()
        non_streaming_counter = _ChatModelStartCounter()
        await agent.ainvoke(
            {"messages": [HumanMessage("weather in paris")]},
            config={"callbacks": [streaming_counter, non_streaming_counter]},
        )

        assert streaming_counter.count == 1
        assert non_streaming_counter.count == 2

    def test_sync_end_to_end_no_leak_in_messages_stream(self) -> None:
        """End-to-end test that `stream_mode="messages"` doesn't leak internal tokens.
        """
        tool_selection_model = _StreamLeakDetectingModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "ToolSelectionResponse",
                                "id": "1",
                                "args": {"tools": ["get_weather"]},
                            }
                        ],
                    ),
                ]
            )
        )
        model = FakeModel(messages=iter([AIMessage(content="Final answer content")]))
        tool_selector = LLMToolSelectorMiddleware(max_tools=1, model=tool_selection_model)

        agent = create_agent(
            model=model,
            tools=[get_weather, search_web],
            middleware=[tool_selector],
        )

        collected_text = ""
        for chunk, _metadata in agent.stream(
            {"messages": [HumanMessage("weather in paris")]}, stream_mode="messages"
        ):
            if isinstance(chunk, AIMessageChunk) and isinstance(chunk.content, str):
                collected_text += chunk.content

        assert tool_selection_model.stream_was_called is False
        assert "Final answer content" in collected_text

    async def test_async_end_to_end_no_leak_in_messages_stream(self) -> None:
        """Async counterpart of the sync end-to-end leak-isolation test."""
        tool_selection_model = _StreamLeakDetectingModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "ToolSelectionResponse",
                                "id": "1",
                                "args": {"tools": ["get_weather"]},
                            }
                        ],
                    ),
                ]
            )
        )
        model = FakeModel(messages=iter([AIMessage(content="Final answer content")]))
        tool_selector = LLMToolSelectorMiddleware(max_tools=1, model=tool_selection_model)

        agent = create_agent(
            model=model,
            tools=[get_weather, search_web],
            middleware=[tool_selector],
        )

        collected_text = ""
        async for chunk, _metadata in agent.astream(
            {"messages": [HumanMessage("weather in paris")]}, stream_mode="messages"
        ):
            if isinstance(chunk, AIMessageChunk) and isinstance(chunk.content, str):
                collected_text += chunk.content

        assert tool_selection_model.stream_was_called is False
        assert "Final answer content" in collected_text


class TestLLMToolSelectorBasic:
    """Test basic tool selection functionality."""

    def test_sync_basic_selection(self) -> None:
        """Test synchronous tool selection."""
        # First call: selector picks tools
        # Second call: agent uses selected tools

        model_requests = []

        @wrap_model_call
        def trace_model_requests(
            request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
        ) -> ModelResponse:
            """Middleware to select relevant tools based on state/context."""
            # Select a small, relevant subset of tools based on state/context
            model_requests.append(request)
            return handler(request)

        tool_selection_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "ToolSelectionResponse",
                                "id": "1",
                                "args": {"tools": ["get_weather", "calculate"]},
                            }
                        ],
                    ),
                ]
            )
        )

        model = FakeModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "get_weather", "id": "2", "args": {"location": "Paris"}}
                        ],
                    ),
                    AIMessage(content="The weather in Paris is 72°F and sunny."),
                ]
            )
        )

        tool_selector = LLMToolSelectorMiddleware(max_tools=2, model=tool_selection_model)

        agent = create_agent(
            model=model,
            tools=[get_weather, search_web, calculate, send_email, get_stock_price],
            middleware=[tool_selector, trace_model_requests],
        )

        response = agent.invoke({"messages": [HumanMessage("What's the weather in Paris?")]})

        assert isinstance(response["messages"][-1], AIMessage)

        for request in model_requests:
            selected_tool_names = []
            for tool_ in request.tools:
                assert isinstance(tool_, BaseTool)
                selected_tool_names.append(tool_.name)
            assert selected_tool_names == ["get_weather", "calculate"]

    async def test_async_basic_selection(self) -> None:
        """Test asynchronous tool selection."""
        tool_selection_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "ToolSelectionResponse",
                                "id": "1",
                                "args": {"tools": ["search_web"]},
                            }
                        ],
                    ),
                ]
            )
        )

        model = FakeModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "search_web", "id": "2", "args": {"query": "Python"}}],
                    ),
                    AIMessage(content="Search results found."),
                ]
            )
        )

        tool_selector = LLMToolSelectorMiddleware(max_tools=1, model=tool_selection_model)

        agent = create_agent(
            model=model,
            tools=[get_weather, search_web, calculate],
            middleware=[tool_selector],
        )

        response = await agent.ainvoke({"messages": [HumanMessage("Search for Python tutorials")]})

        assert isinstance(response["messages"][-1], AIMessage)


class TestMaxToolsLimiting:
    """Test max_tools limiting behavior."""

    def test_max_tools_limits_selection(self) -> None:
        """Test that max_tools limits selection when model selects too many tools."""
        model_requests = []

        @wrap_model_call
        def trace_model_requests(
            request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
        ) -> ModelResponse:
            model_requests.append(request)
            return handler(request)

        # Selector model tries to select 4 tools
        tool_selection_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "ToolSelectionResponse",
                                "id": "1",
                                "args": {
                                    "tools": [
                                        "get_weather",
                                        "search_web",
                                        "calculate",
                                        "send_email",
                                    ]
                                },
                            }
                        ],
                    ),
                ]
            )
        )

        model = FakeModel(messages=iter([AIMessage(content="Done")]))

        # But max_tools=2, so only first 2 should be used
        tool_selector = LLMToolSelectorMiddleware(max_tools=2, model=tool_selection_model)

        agent = create_agent(
            model=model,
            tools=[get_weather, search_web, calculate, send_email],
            middleware=[tool_selector, trace_model_requests],
        )

        agent.invoke({"messages": [HumanMessage("test")]})

        # Verify only 2 tools were passed to the main model
        assert len(model_requests) > 0
        for request in model_requests:
            assert len(request.tools) == 2
            tool_names = []
            for tool_ in request.tools:
                assert isinstance(tool_, BaseTool)
                tool_names.append(tool_.name)
            # Should be first 2 from the selection
            assert tool_names == ["get_weather", "search_web"]

    def test_no_max_tools_uses_all_selected(self) -> None:
        """Test that when max_tools is None, all selected tools are used."""
        model_requests = []

        @wrap_model_call
        def trace_model_requests(
            request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
        ) -> ModelResponse:
            model_requests.append(request)
            return handler(request)

        tool_selection_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "ToolSelectionResponse",
                                "id": "1",
                                "args": {
                                    "tools": [
                                        "get_weather",
                                        "search_web",
                                        "calculate",
                                        "get_stock_price",
                                    ]
                                },
                            }
                        ],
                    ),
                ]
            )
        )

        model = FakeModel(messages=iter([AIMessage(content="Done")]))

        # No max_tools specified
        tool_selector = LLMToolSelectorMiddleware(model=tool_selection_model)

        agent = create_agent(
            model=model,
            tools=[get_weather, search_web, calculate, send_email, get_stock_price],
            middleware=[tool_selector, trace_model_requests],
        )

        agent.invoke({"messages": [HumanMessage("test")]})

        # All 4 selected tools should be present
        assert len(model_requests) > 0
        for request in model_requests:
            assert len(request.tools) == 4
            tool_names = []
            for tool_ in request.tools:
                assert isinstance(tool_, BaseTool)
                tool_names.append(tool_.name)
            assert set(tool_names) == {
                "get_weather",
                "search_web",
                "calculate",
                "get_stock_price",
            }


class TestAlwaysInclude:
    """Test always_include functionality."""

    def test_always_include_tools_present(self) -> None:
        """Test that always_include tools are always present in the request."""
        model_requests = []

        @wrap_model_call
        def trace_model_requests(
            request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
        ) -> ModelResponse:
            model_requests.append(request)
            return handler(request)

        # Selector picks only search_web
        tool_selection_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "ToolSelectionResponse",
                                "id": "1",
                                "args": {"tools": ["search_web"]},
                            }
                        ],
                    ),
                ]
            )
        )

        model = FakeModel(messages=iter([AIMessage(content="Done")]))

        # But send_email is always included
        tool_selector = LLMToolSelectorMiddleware(
            max_tools=1, always_include=["send_email"], model=tool_selection_model
        )

        agent = create_agent(
            model=model,
            tools=[get_weather, search_web, send_email],
            middleware=[tool_selector, trace_model_requests],
        )

        agent.invoke({"messages": [HumanMessage("test")]})

        # Both selected and always_include tools should be present
        assert len(model_requests) > 0
        for request in model_requests:
            tool_names = []
            for tool_ in request.tools:
                assert isinstance(tool_, BaseTool)
                tool_names.append(tool_.name)
            assert "search_web" in tool_names
            assert "send_email" in tool_names
            assert len(tool_names) == 2

    def test_always_include_not_counted_against_max(self) -> None:
        """Test that always_include tools don't count against max_tools limit."""
        model_requests = []

        @wrap_model_call
        def trace_model_requests(
            request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
        ) -> ModelResponse:
            model_requests.append(request)
            return handler(request)

        # Selector picks 2 tools
        tool_selection_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "ToolSelectionResponse",
                                "id": "1",
                                "args": {"tools": ["get_weather", "search_web"]},
                            }
                        ],
                    ),
                ]
            )
        )

        model = FakeModel(messages=iter([AIMessage(content="Done")]))

        # max_tools=2, but we also have 2 always_include tools
        tool_selector = LLMToolSelectorMiddleware(
            max_tools=2,
            always_include=["send_email", "calculate"],
            model=tool_selection_model,
        )

        agent = create_agent(
            model=model,
            tools=[get_weather, search_web, calculate, send_email],
            middleware=[tool_selector, trace_model_requests],
        )

        agent.invoke({"messages": [HumanMessage("test")]})

        # Should have 2 selected + 2 always_include = 4 total
        assert len(model_requests) > 0
        for request in model_requests:
            assert len(request.tools) == 4
            tool_names = []
            for tool_ in request.tools:
                assert isinstance(tool_, BaseTool)
                tool_names.append(tool_.name)
            assert "get_weather" in tool_names
            assert "search_web" in tool_names
            assert "send_email" in tool_names
            assert "calculate" in tool_names

    def test_multiple_always_include_tools(self) -> None:
        """Test that multiple always_include tools are all present."""
        model_requests = []

        @wrap_model_call
        def trace_model_requests(
            request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
        ) -> ModelResponse:
            model_requests.append(request)
            return handler(request)

        # Selector picks 1 tool
        tool_selection_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "ToolSelectionResponse",
                                "id": "1",
                                "args": {"tools": ["get_weather"]},
                            }
                        ],
                    ),
                ]
            )
        )

        model = FakeModel(messages=iter([AIMessage(content="Done")]))

        tool_selector = LLMToolSelectorMiddleware(
            max_tools=1,
            always_include=["send_email", "calculate", "get_stock_price"],
            model=tool_selection_model,
        )

        agent = create_agent(
            model=model,
            tools=[get_weather, search_web, send_email, calculate, get_stock_price],
            middleware=[tool_selector, trace_model_requests],
        )

        agent.invoke({"messages": [HumanMessage("test")]})

        # Should have 1 selected + 3 always_include = 4 total
        assert len(model_requests) > 0
        for request in model_requests:
            assert len(request.tools) == 4
            tool_names = []
            for tool_ in request.tools:
                assert isinstance(tool_, BaseTool)
                tool_names.append(tool_.name)
            assert "get_weather" in tool_names
            assert "send_email" in tool_names
            assert "calculate" in tool_names
            assert "get_stock_price" in tool_names


class TestDuplicateAndInvalidTools:
    """Test handling of duplicate and invalid tool selections."""

    def test_duplicate_tool_selection_deduplicated(self) -> None:
        """Test that duplicate tool selections are deduplicated."""
        model_requests = []

        @wrap_model_call
        def trace_model_requests(
            request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
        ) -> ModelResponse:
            model_requests.append(request)
            return handler(request)

        # Selector returns duplicates
        tool_selection_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "ToolSelectionResponse",
                                "id": "1",
                                "args": {
                                    "tools": [
                                        "get_weather",
                                        "get_weather",
                                        "search_web",
                                        "search_web",
                                    ]
                                },
                            }
                        ],
                    ),
                ]
            )
        )

        model = FakeModel(messages=iter([AIMessage(content="Done")]))

        tool_selector = LLMToolSelectorMiddleware(max_tools=5, model=tool_selection_model)

        agent = create_agent(
            model=model,
            tools=[get_weather, search_web, calculate],
            middleware=[tool_selector, trace_model_requests],
        )

        agent.invoke({"messages": [HumanMessage("test")]})

        # Duplicates should be removed
        assert len(model_requests) > 0
        for request in model_requests:
            tool_names = []
            for tool_ in request.tools:
                assert isinstance(tool_, BaseTool)
                tool_names.append(tool_.name)
            assert tool_names == ["get_weather", "search_web"]
            assert len(tool_names) == 2

    def test_max_tools_with_duplicates(self) -> None:
        """Test that max_tools works correctly with duplicate selections."""
        model_requests: list[ModelRequest] = []

        @wrap_model_call
        def trace_model_requests(
            request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
        ) -> ModelResponse:
            model_requests.append(request)
            return handler(request)

        # Selector returns duplicates but max_tools=2
        tool_selection_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "ToolSelectionResponse",
                                "id": "1",
                                "args": {
                                    "tools": [
                                        "get_weather",
                                        "get_weather",
                                        "search_web",
                                        "search_web",
                                        "calculate",
                                    ]
                                },
                            }
                        ],
                    ),
                ]
            )
        )

        model = FakeModel(messages=iter([AIMessage(content="Done")]))

        tool_selector = LLMToolSelectorMiddleware(max_tools=2, model=tool_selection_model)

        agent = create_agent(
            model=model,
            tools=[get_weather, search_web, calculate],
            middleware=[tool_selector, trace_model_requests],
        )

        agent.invoke({"messages": [HumanMessage("test")]})

        # Should deduplicate and respect max_tools
        assert len(model_requests) > 0
        for request in model_requests:
            tool_names = []
            for tool_ in request.tools:
                assert isinstance(tool_, BaseTool)
                tool_names.append(tool_.name)
            assert len(tool_names) == 2
            assert "get_weather" in tool_names
            assert "search_web" in tool_names


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_tools_list_raises_error(self) -> None:
        """Test that empty tools list raises an error in schema creation."""
        with pytest.raises(AssertionError, match="tools must be non-empty"):
            _create_tool_selection_response([])

    def test_malformed_selection_response_missing_tools_key(self) -> None:
        """Test that a structured output response missing `tools` doesn't raise `KeyError`."""
        middleware = LLMToolSelectorMiddleware()
        request = ModelRequest(
            model=None,  # type: ignore[arg-type]
            messages=[],
            tools=[get_weather, search_web],
            state={},
            runtime=None,  # type: ignore[arg-type]
        )

        modified_request = middleware._process_selection_response(
            {},  # malformed: missing the expected "tools" key
            available_tools=[get_weather, search_web],
            valid_tool_names=["get_weather", "search_web"],
            request=request,
        )

        assert modified_request.tools == []

    def test_sync_retries_once_then_succeeds_on_malformed_response(self) -> None:
        """A single malformed response should be retried and recover on the second try."""
        tool_selection_model = FakeModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "ToolSelectionResponse", "id": "1", "args": {}},
                        ],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "ToolSelectionResponse",
                                "id": "2",
                                "args": {"tools": ["get_weather"]},
                            }
                        ],
                    ),
                ]
            )
        )
        model = FakeModel(messages=iter([AIMessage(content="Done")]))
        tool_selector = LLMToolSelectorMiddleware(max_tools=1, model=tool_selection_model)

        agent = create_agent(
            model=model,
            tools=[get_weather, search_web],
            middleware=[tool_selector],
        )

        response = agent.invoke({"messages": [HumanMessage("test")]})

        assert isinstance(response["messages"][-1], AIMessage)

    def test_sync_raises_after_malformed_response_twice(self) -> None:
        """Two malformed responses in a row should raise a clear `ValueError`."""
        tool_selection_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "ToolSelectionResponse", "id": "1", "args": {}},
                        ],
                    ),
                ]
            )
        )
        model = FakeModel(messages=iter([AIMessage(content="Done")]))
        tool_selector = LLMToolSelectorMiddleware(max_tools=1, model=tool_selection_model)

        agent = create_agent(
            model=model,
            tools=[get_weather, search_web],
            middleware=[tool_selector],
        )

        with pytest.raises(ValueError, match="malformed"):
            agent.invoke({"messages": [HumanMessage("test")]})

    async def test_async_raises_after_malformed_response_twice(self) -> None:
        """Two malformed responses in a row should raise a clear `ValueError`."""
        tool_selection_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "ToolSelectionResponse", "id": "1", "args": {}},
                        ],
                    ),
                ]
            )
        )
        model = FakeModel(messages=iter([AIMessage(content="Done")]))
        tool_selector = LLMToolSelectorMiddleware(max_tools=1, model=tool_selection_model)

        agent = create_agent(
            model=model,
            tools=[get_weather, search_web],
            middleware=[tool_selector],
        )

        with pytest.raises(ValueError, match="malformed"):
            await agent.ainvoke({"messages": [HumanMessage("test")]})
