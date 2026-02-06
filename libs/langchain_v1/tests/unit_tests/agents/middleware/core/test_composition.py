"""Unit tests for _chain_model_call_handlers handler composition."""

from collections.abc import Callable
from typing import Any, TypedDict, cast

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from langchain.agents import AgentState
from langchain.agents.factory import _chain_model_call_handlers, _ComposedExtendedModelResponse
from langchain.agents.middleware.types import ExtendedModelResponse, ModelRequest, ModelResponse


def create_test_request(**kwargs: Any) -> ModelRequest:
    """Helper to create a `ModelRequest` with sensible defaults."""
    defaults: dict[str, Any] = {
        "messages": [],
        "model": None,
        "system_prompt": None,
        "tool_choice": None,
        "tools": [],
        "response_format": None,
        "state": {},
        "runtime": cast("Runtime", object()),
    }
    defaults.update(kwargs)
    return ModelRequest(**defaults)


def create_mock_base_handler(content: str = "test") -> Callable[[ModelRequest], ModelResponse]:
    """Helper to create a base handler that returns `ModelResponse`."""

    def mock_base_handler(_: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content=content)], structured_response=None)

    return mock_base_handler


class TestChainModelCallHandlers:
    """Test the `_chain_model_call_handlers` composition function."""

    def test_empty_handlers_returns_none(self) -> None:
        """Test that empty handlers list returns None."""
        result = _chain_model_call_handlers([])
        assert result is None

    def test_single_handler_returns_unchanged(self) -> None:
        """Test that single handler is wrapped to normalize output."""

        def handler(
            request: ModelRequest, base_handler: Callable[[ModelRequest], ModelResponse]
        ) -> ModelResponse:
            return base_handler(request)

        result = _chain_model_call_handlers([handler])
        # Result is wrapped to normalize, so it won't be identical
        assert result is not None
        assert callable(result)

    def test_two_handlers_basic_composition(self) -> None:
        """Test basic composition of two handlers."""
        execution_order = []

        def outer(
            request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
        ) -> ModelResponse:
            execution_order.append("outer-before")
            result = handler(request)
            execution_order.append("outer-after")
            return result

        def inner(
            request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
        ) -> ModelResponse:
            execution_order.append("inner-before")
            result = handler(request)
            execution_order.append("inner-after")
            return result

        composed = _chain_model_call_handlers([outer, inner])
        assert composed is not None

        result = composed(create_test_request(), create_mock_base_handler())

        assert execution_order == [
            "outer-before",
            "inner-before",
            "inner-after",
            "outer-after",
        ]
        # Outermost result is always _ComposedExtendedModelResponse
        assert isinstance(result, _ComposedExtendedModelResponse)
        assert result.model_response.result[0].content == "test"

    def test_two_handlers_with_commands(self) -> None:
        """Test that commands from inner and outer are collected correctly."""

        def outer(
            request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
        ) -> ExtendedModelResponse:
            response = handler(request)
            return ExtendedModelResponse(
                model_response=response,
                command=Command(update={"outer_key": "outer_val"}),
            )

        def inner(
            request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
        ) -> ExtendedModelResponse:
            response = handler(request)
            return ExtendedModelResponse(
                model_response=response,
                command=Command(update={"inner_key": "inner_val"}),
            )

        composed = _chain_model_call_handlers([outer, inner])
        assert composed is not None

        result = composed(create_test_request(), create_mock_base_handler())

        assert isinstance(result, _ComposedExtendedModelResponse)
        # Commands are collected: inner first, then outer
        assert len(result.commands) == 2
        assert result.commands[0].update == {"inner_key": "inner_val"}
        assert result.commands[1].update == {"outer_key": "outer_val"}

    def test_three_handlers_composition(self) -> None:
        """Test composition of three handlers."""
        execution_order = []

        def first(
            request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
        ) -> ModelResponse:
            execution_order.append("first-before")
            result = handler(request)
            execution_order.append("first-after")
            return result

        def second(
            request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
        ) -> ModelResponse:
            execution_order.append("second-before")
            result = handler(request)
            execution_order.append("second-after")
            return result

        def third(
            request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
        ) -> ModelResponse:
            execution_order.append("third-before")
            result = handler(request)
            execution_order.append("third-after")
            return result

        composed = _chain_model_call_handlers([first, second, third])
        assert composed is not None

        result = composed(create_test_request(), create_mock_base_handler())

        # First wraps second wraps third
        assert execution_order == [
            "first-before",
            "second-before",
            "third-before",
            "third-after",
            "second-after",
            "first-after",
        ]
        assert isinstance(result, _ComposedExtendedModelResponse)
        assert result.model_response.result[0].content == "test"

    def test_inner_handler_retry(self) -> None:
        """Test inner handler retrying before outer sees response."""
        inner_attempts = []

        def outer_passthrough(
            request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
        ) -> ModelResponse:
            return handler(request)

        def inner_with_retry(
            request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
        ) -> ModelResponse | AIMessage:
            for attempt in range(3):
                inner_attempts.append(attempt)
                try:
                    return handler(request)
                except ValueError:
                    if attempt == 2:
                        raise
            return AIMessage(content="should not reach")

        composed = _chain_model_call_handlers([outer_passthrough, inner_with_retry])
        assert composed is not None

        call_count = {"value": 0}

        def mock_base_handler(_: ModelRequest) -> ModelResponse:
            call_count["value"] += 1
            if call_count["value"] < 3:
                msg = "fail"
                raise ValueError(msg)
            return ModelResponse(result=[AIMessage(content="success")], structured_response=None)

        result = composed(create_test_request(), mock_base_handler)

        assert inner_attempts == [0, 1, 2]
        assert isinstance(result, _ComposedExtendedModelResponse)
        assert result.model_response.result[0].content == "success"

    def test_error_to_success_conversion(self) -> None:
        """Test handler converting error to success response."""

        def outer_error_handler(
            request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
        ) -> ModelResponse | AIMessage:
            try:
                return handler(request)
            except Exception:
                # Middleware can return AIMessage - it will be normalized to ModelResponse
                return AIMessage(content="Fallback response")

        def inner_passthrough(
            request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
        ) -> ModelResponse:
            return handler(request)

        composed = _chain_model_call_handlers([outer_error_handler, inner_passthrough])
        assert composed is not None

        def mock_base_handler(_: ModelRequest) -> ModelResponse:
            msg = "Model failed"
            raise ValueError(msg)

        result = composed(create_test_request(), mock_base_handler)

        # AIMessage was automatically normalized into ExtendedModelResponse
        assert isinstance(result, _ComposedExtendedModelResponse)
        assert result.model_response.result[0].content == "Fallback response"
        assert result.model_response.structured_response is None

    def test_request_modification(self) -> None:
        """Test handlers modifying the request."""
        requests_seen = []

        def outer_add_context(
            request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
        ) -> ModelResponse:
            modified_request = create_test_request(
                messages=[*request.messages], system_prompt="Added by outer"
            )
            return handler(modified_request)

        def inner_track_request(
            request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
        ) -> ModelResponse:
            requests_seen.append(request.system_prompt)
            return handler(request)

        composed = _chain_model_call_handlers([outer_add_context, inner_track_request])
        assert composed is not None

        result = composed(create_test_request(), create_mock_base_handler(content="response"))

        assert requests_seen == ["Added by outer"]
        assert isinstance(result, _ComposedExtendedModelResponse)
        assert result.model_response.result[0].content == "response"

    def test_composition_preserves_state_and_runtime(self) -> None:
        """Test that state and runtime are passed through composition."""

        class CustomState(AgentState[Any]):
            test: str

        class CustomContext(TypedDict):
            test: str

        state_values = []
        runtime_values = []

        def outer(
            request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
        ) -> ModelResponse:
            state_values.append(("outer", request.state))
            runtime_values.append(("outer", request.runtime))
            return handler(request)

        def inner(
            request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
        ) -> ModelResponse:
            state_values.append(("inner", request.state))
            runtime_values.append(("inner", request.runtime))
            return handler(request)

        composed = _chain_model_call_handlers([outer, inner])
        assert composed is not None

        test_state = CustomState(messages=[], test="state")
        test_runtime = Runtime(context=CustomContext(test="runtime"))

        # Create request with state and runtime
        test_request = create_test_request(state=test_state, runtime=test_runtime)
        result = composed(test_request, create_mock_base_handler())

        # Both handlers should see same state and runtime
        assert state_values == [("outer", test_state), ("inner", test_state)]
        assert runtime_values == [("outer", test_runtime), ("inner", test_runtime)]
        assert isinstance(result, _ComposedExtendedModelResponse)
        assert result.model_response.result[0].content == "test"

    def test_multiple_yields_in_retry_loop(self) -> None:
        """Test handler that retries multiple times."""
        call_count = {"value": 0}

        def outer_counts_calls(
            request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
        ) -> ModelResponse:
            call_count["value"] += 1
            return handler(request)

        def inner_retries(
            request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
        ) -> ModelResponse:
            try:
                return handler(request)
            except ValueError:
                # Retry once on error
                return handler(request)

        composed = _chain_model_call_handlers([outer_counts_calls, inner_retries])
        assert composed is not None

        attempt = {"value": 0}

        def mock_base_handler(_: ModelRequest) -> ModelResponse:
            attempt["value"] += 1
            if attempt["value"] == 1:
                msg = "fail"
                raise ValueError(msg)
            return ModelResponse(result=[AIMessage(content="ok")], structured_response=None)

        result = composed(create_test_request(), mock_base_handler)

        # Outer called once, inner retried so base handler called twice
        assert call_count["value"] == 1
        assert attempt["value"] == 2
        assert isinstance(result, _ComposedExtendedModelResponse)
        assert result.model_response.result[0].content == "ok"
