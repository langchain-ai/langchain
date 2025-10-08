"""Unit tests for _chain_model_call_handlers handler composition."""

import pytest
from langchain_core.messages import AIMessage

from langchain.agents.factory import _chain_model_call_handlers
from langchain.agents.middleware.types import ModelRequest


def create_test_request(**kwargs):
    """Helper to create a ModelRequest with sensible defaults."""
    defaults = {
        "messages": [],
        "model": None,
        "system_prompt": None,
        "tool_choice": None,
        "tools": [],
        "response_format": None,
    }
    defaults.update(kwargs)
    return ModelRequest(**defaults)


class TestChainModelCallHandlers:
    """Test the _chain_model_call_handlers composition function."""

    def test_empty_handlers_returns_none(self) -> None:
        """Test that empty handlers list returns None."""
        result = _chain_model_call_handlers([])
        assert result is None

    def test_single_handler_returns_unchanged(self) -> None:
        """Test that single handler is returned without composition."""

        def handler(request, state, runtime, base_handler):
            return base_handler(request)

        result = _chain_model_call_handlers([handler])
        assert result is handler

    def test_two_handlers_basic_composition(self) -> None:
        """Test basic composition of two handlers."""
        execution_order = []

        def outer(request, state, runtime, handler):
            execution_order.append("outer-before")
            result = handler(request)
            execution_order.append("outer-after")
            return result

        def inner(request, state, runtime, handler):
            execution_order.append("inner-before")
            result = handler(request)
            execution_order.append("inner-after")
            return result

        composed = _chain_model_call_handlers([outer, inner])
        assert composed is not None

        # Execute the composed handler
        def mock_base_handler(req):
            return AIMessage(content="test")

        result = composed(create_test_request(), None, None, mock_base_handler)

        assert execution_order == [
            "outer-before",
            "inner-before",
            "inner-after",
            "outer-after",
        ]
        assert result.content == "test"

    def test_three_handlers_composition(self) -> None:
        """Test composition of three handlers."""
        execution_order = []

        def first(request, state, runtime, handler):
            execution_order.append("first-before")
            result = handler(request)
            execution_order.append("first-after")
            return result

        def second(request, state, runtime, handler):
            execution_order.append("second-before")
            result = handler(request)
            execution_order.append("second-after")
            return result

        def third(request, state, runtime, handler):
            execution_order.append("third-before")
            result = handler(request)
            execution_order.append("third-after")
            return result

        composed = _chain_model_call_handlers([first, second, third])
        assert composed is not None

        def mock_base_handler(req):
            return AIMessage(content="test")

        result = composed(create_test_request(), None, None, mock_base_handler)

        # First wraps second wraps third
        assert execution_order == [
            "first-before",
            "second-before",
            "third-before",
            "third-after",
            "second-after",
            "first-after",
        ]
        assert result.content == "test"

    def test_inner_handler_retry(self) -> None:
        """Test inner handler retrying before outer sees response."""
        inner_attempts = []

        def outer_passthrough(request, state, runtime, handler):
            return handler(request)

        def inner_with_retry(request, state, runtime, handler):
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

        def mock_base_handler(req):
            call_count["value"] += 1
            if call_count["value"] < 3:
                raise ValueError("fail")
            return AIMessage(content="success")

        result = composed(create_test_request(), None, None, mock_base_handler)

        assert inner_attempts == [0, 1, 2]
        assert result.content == "success"

    def test_error_to_success_conversion(self) -> None:
        """Test handler converting error to success response."""

        def outer_error_handler(request, state, runtime, handler):
            try:
                return handler(request)
            except Exception:
                return AIMessage(content="Fallback response")

        def inner_passthrough(request, state, runtime, handler):
            return handler(request)

        composed = _chain_model_call_handlers([outer_error_handler, inner_passthrough])
        assert composed is not None

        def mock_base_handler(req):
            raise ValueError("Model failed")

        result = composed(create_test_request(), None, None, mock_base_handler)

        assert result.content == "Fallback response"

    def test_request_modification(self) -> None:
        """Test handlers modifying the request."""
        requests_seen = []

        def outer_add_context(request, state, runtime, handler):
            modified_request = create_test_request(
                messages=[*request.messages], system_prompt="Added by outer"
            )
            return handler(modified_request)

        def inner_track_request(request, state, runtime, handler):
            requests_seen.append(request.system_prompt)
            return handler(request)

        composed = _chain_model_call_handlers([outer_add_context, inner_track_request])
        assert composed is not None

        def mock_base_handler(req):
            return AIMessage(content="response")

        result = composed(create_test_request(), None, None, mock_base_handler)

        assert requests_seen == ["Added by outer"]
        assert result.content == "response"

    def test_composition_preserves_state_and_runtime(self) -> None:
        """Test that state and runtime are passed through composition."""
        state_values = []
        runtime_values = []

        def outer(request, state, runtime, handler):
            state_values.append(("outer", state))
            runtime_values.append(("outer", runtime))
            return handler(request)

        def inner(request, state, runtime, handler):
            state_values.append(("inner", state))
            runtime_values.append(("inner", runtime))
            return handler(request)

        composed = _chain_model_call_handlers([outer, inner])
        assert composed is not None

        test_state = {"test": "state"}
        test_runtime = {"test": "runtime"}

        def mock_base_handler(req):
            return AIMessage(content="test")

        result = composed(create_test_request(), test_state, test_runtime, mock_base_handler)

        # Both handlers should see same state and runtime
        assert state_values == [("outer", test_state), ("inner", test_state)]
        assert runtime_values == [("outer", test_runtime), ("inner", test_runtime)]
        assert result.content == "test"

    def test_multiple_yields_in_retry_loop(self) -> None:
        """Test handler that retries multiple times."""
        call_count = {"value": 0}

        def outer_counts_calls(request, state, runtime, handler):
            call_count["value"] += 1
            return handler(request)

        def inner_retries(request, state, runtime, handler):
            try:
                return handler(request)
            except ValueError:
                # Retry once on error
                return handler(request)

        composed = _chain_model_call_handlers([outer_counts_calls, inner_retries])
        assert composed is not None

        attempt = {"value": 0}

        def mock_base_handler(req):
            attempt["value"] += 1
            if attempt["value"] == 1:
                raise ValueError("fail")
            return AIMessage(content="ok")

        result = composed(create_test_request(), None, None, mock_base_handler)

        # Outer called once, inner retried so base handler called twice
        assert call_count["value"] == 1
        assert attempt["value"] == 2
        assert result.content == "ok"
