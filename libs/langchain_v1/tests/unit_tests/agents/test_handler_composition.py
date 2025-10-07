"""Unit tests for _chain_model_call_handlers generator composition."""

import pytest
from langchain_core.messages import AIMessage

from langchain.agents.factory import _chain_model_call_handlers
from langchain.agents.middleware.types import ModelRequest, ModelResponse


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

        def handler(request, state, runtime):
            response = yield request
            return response

        result = _chain_model_call_handlers([handler])
        assert result is handler

    def test_two_handlers_basic_composition(self) -> None:
        """Test basic composition of two handlers."""
        execution_order = []

        def outer(request, state, runtime):
            execution_order.append("outer-before")
            response = yield request
            execution_order.append("outer-after")
            return response

        def inner(request, state, runtime):
            execution_order.append("inner-before")
            response = yield request
            execution_order.append("inner-after")
            return response

        composed = _chain_model_call_handlers([outer, inner])
        assert composed is not None

        # Execute the composed handler
        gen = composed(create_test_request(), None, None)
        request = next(gen)

        # Simulate model response
        model_response = ModelResponse(action="return", result=AIMessage(content="test"))

        try:
            gen.send(model_response)
        except StopIteration as e:
            final_response = e.value

        assert execution_order == [
            "outer-before",
            "inner-before",
            "inner-after",
            "outer-after",
        ]
        assert final_response.action == "return"

    def test_three_handlers_composition(self) -> None:
        """Test composition of three handlers."""
        execution_order = []

        def first(request, state, runtime):
            execution_order.append("first-before")
            response = yield request
            execution_order.append("first-after")
            return response

        def second(request, state, runtime):
            execution_order.append("second-before")
            response = yield request
            execution_order.append("second-after")
            return response

        def third(request, state, runtime):
            execution_order.append("third-before")
            response = yield request
            execution_order.append("third-after")
            return response

        composed = _chain_model_call_handlers([first, second, third])
        assert composed is not None

        gen = composed(create_test_request(), None, None)
        request = next(gen)

        model_response = ModelResponse(action="return", result=AIMessage(content="test"))

        try:
            gen.send(model_response)
        except StopIteration as e:
            final_response = e.value

        # First wraps second wraps third
        assert execution_order == [
            "first-before",
            "second-before",
            "third-before",
            "third-after",
            "second-after",
            "first-after",
        ]
        assert final_response.action == "return"

    def test_outer_handler_retry(self) -> None:
        """Test outer handler retrying after error."""
        call_count = {"value": 0}

        def outer_with_retry(request, state, runtime):
            for _ in range(3):
                response = yield request
                if response.action == "return":
                    return response
            return response

        def inner_passthrough(request, state, runtime):
            call_count["value"] += 1
            response = yield request
            return response

        composed = _chain_model_call_handlers([outer_with_retry, inner_passthrough])
        assert composed is not None

        gen = composed(create_test_request(), None, None)

        # First attempt - error
        request = next(gen)
        error_response = ModelResponse(action="raise", exception=ValueError("First attempt fails"))
        request = gen.send(error_response)

        # Second attempt - success
        success_response = ModelResponse(action="return", result=AIMessage(content="success"))

        try:
            gen.send(success_response)
        except StopIteration as e:
            final_response = e.value

        assert call_count["value"] == 2
        assert final_response.action == "return"
        assert final_response.result.content == "success"

    def test_inner_handler_retry(self) -> None:
        """Test inner handler retrying before outer sees response."""
        inner_attempts = []

        def outer_passthrough(request, state, runtime):
            response = yield request
            return response

        def inner_with_retry(request, state, runtime):
            for attempt in range(3):
                inner_attempts.append(attempt)
                response = yield request
                if response.action == "return":
                    return response
            return response

        composed = _chain_model_call_handlers([outer_passthrough, inner_with_retry])
        assert composed is not None

        gen = composed(create_test_request(), None, None)

        # Inner will retry twice before success
        request = next(gen)
        error_response = ModelResponse(action="raise", exception=ValueError("fail"))
        request = gen.send(error_response)

        request = gen.send(error_response)

        success_response = ModelResponse(action="return", result=AIMessage(content="success"))

        try:
            gen.send(success_response)
        except StopIteration as e:
            final_response = e.value

        assert inner_attempts == [0, 1, 2]
        assert final_response.action == "return"

    def test_response_modification_by_outer(self) -> None:
        """Test outer handler modifying response from inner."""

        def outer_uppercase(request, state, runtime):
            response = yield request
            if response.action == "return" and response.result:
                modified = AIMessage(content=response.result.content.upper())
                response = ModelResponse(action="return", result=modified)
            return response

        def inner_passthrough(request, state, runtime):
            response = yield request
            return response

        composed = _chain_model_call_handlers([outer_uppercase, inner_passthrough])
        assert composed is not None

        gen = composed(create_test_request(), None, None)
        request = next(gen)

        model_response = ModelResponse(action="return", result=AIMessage(content="hello world"))

        try:
            gen.send(model_response)
        except StopIteration as e:
            final_response = e.value

        assert final_response.action == "return"
        assert final_response.result.content == "HELLO WORLD"

    def test_error_to_success_conversion(self) -> None:
        """Test handler converting error to success response."""

        def outer_error_handler(request, state, runtime):
            response = yield request
            if response.action == "raise":
                fallback = AIMessage(content="Fallback response")
                response = ModelResponse(action="return", result=fallback)
            return response

        def inner_passthrough(request, state, runtime):
            response = yield request
            return response

        composed = _chain_model_call_handlers([outer_error_handler, inner_passthrough])
        assert composed is not None

        gen = composed(create_test_request(), None, None)
        request = next(gen)

        error_response = ModelResponse(action="raise", exception=ValueError("Model failed"))

        try:
            gen.send(error_response)
        except StopIteration as e:
            final_response = e.value

        assert final_response.action == "return"
        assert final_response.result.content == "Fallback response"

    def test_request_modification(self) -> None:
        """Test handlers modifying the request."""
        requests_seen = []

        def outer_add_context(request, state, runtime):
            modified_request = create_test_request(
                messages=[*request.messages], system_prompt="Added by outer"
            )
            response = yield modified_request
            return response

        def inner_track_request(request, state, runtime):
            requests_seen.append(request.system_prompt)
            response = yield request
            return response

        composed = _chain_model_call_handlers([outer_add_context, inner_track_request])
        assert composed is not None

        gen = composed(create_test_request(), None, None)
        request = next(gen)

        model_response = ModelResponse(action="return", result=AIMessage(content="response"))

        try:
            gen.send(model_response)
        except StopIteration:
            pass

        assert requests_seen == ["Added by outer"]

    def test_handler_immediate_return(self) -> None:
        """Test handler that returns immediately without yielding."""

        def outer_immediate_return(request, state, runtime):
            # Return cached response without calling inner
            return ModelResponse(action="return", result=AIMessage(content="cached"))
            yield  # Make it a generator (unreachable)

        def inner_never_called(request, state, runtime):
            pytest.fail("Inner handler should not be called")
            response = yield request
            return response

        composed = _chain_model_call_handlers([outer_immediate_return, inner_never_called])
        assert composed is not None

        gen = composed(create_test_request(), None, None)

        try:
            next(gen)
            pytest.fail("Should have returned immediately")
        except StopIteration as e:
            final_response = e.value

        assert final_response.action == "return"
        assert final_response.result.content == "cached"

    def test_inner_immediate_return(self) -> None:
        """Test inner handler returning immediately without yielding."""
        outer_saw_response = []

        def outer_passthrough(request, state, runtime):
            response = yield request
            outer_saw_response.append(response.result.content)
            return response

        def inner_immediate_return(request, state, runtime):
            return ModelResponse(action="return", result=AIMessage(content="immediate"))
            yield  # Make it a generator (unreachable)

        composed = _chain_model_call_handlers([outer_passthrough, inner_immediate_return])
        assert composed is not None

        gen = composed(create_test_request(), None, None)

        # When inner returns immediately, the composition doesn't yield at all.
        # Outer initializes, inner returns immediately, outer receives response
        # and returns, so the composed generator returns on first next().
        try:
            next(gen)
            pytest.fail("Expected StopIteration on immediate return")
        except StopIteration as e:
            final_response = e.value

        # Outer should have seen inner's immediate response
        assert outer_saw_response == ["immediate"]
        assert final_response.result.content == "immediate"

    def test_both_handlers_retry(self) -> None:
        """Test both outer and inner handlers implementing retry logic."""
        outer_attempts = []
        inner_attempts = []

        def outer_retry_twice(request, state, runtime):
            for attempt in range(2):
                outer_attempts.append(attempt)
                response = yield request
                if response.action == "return":
                    return response
            return response

        def inner_retry_twice(request, state, runtime):
            for attempt in range(2):
                inner_attempts.append(attempt)
                response = yield request
                if response.action == "return":
                    return response
            return response

        composed = _chain_model_call_handlers([outer_retry_twice, inner_retry_twice])
        assert composed is not None

        gen = composed(create_test_request(), None, None)

        # All attempts fail
        for _ in range(4):  # 2 outer * 2 inner
            request = next(gen) if _ == 0 else gen.send(error)
            error = ModelResponse(action="raise", exception=ValueError("fail"))

        try:
            gen.send(error)
        except StopIteration as e:
            final_response = e.value

        # Outer retries (0, 1), each triggers inner retries (0, 1)
        assert len(outer_attempts) == 2
        assert len(inner_attempts) == 4  # 2 outer * 2 inner
        assert final_response.action == "raise"

    def test_composition_preserves_state_and_runtime(self) -> None:
        """Test that state and runtime are passed through composition."""
        state_values = []
        runtime_values = []

        def outer(request, state, runtime):
            state_values.append(("outer", state))
            runtime_values.append(("outer", runtime))
            response = yield request
            return response

        def inner(request, state, runtime):
            state_values.append(("inner", state))
            runtime_values.append(("inner", runtime))
            response = yield request
            return response

        composed = _chain_model_call_handlers([outer, inner])
        assert composed is not None

        test_state = {"test": "state"}
        test_runtime = {"test": "runtime"}

        gen = composed(create_test_request(), test_state, test_runtime)
        request = next(gen)

        model_response = ModelResponse(action="return", result=AIMessage(content="test"))

        try:
            gen.send(model_response)
        except StopIteration:
            pass

        # Both handlers should see same state and runtime
        assert state_values == [("outer", test_state), ("inner", test_state)]
        assert runtime_values == [("outer", test_runtime), ("inner", test_runtime)]

    def test_multiple_yields_in_retry_loop(self) -> None:
        """Test handler that yields multiple times in a loop."""
        yield_count = {"value": 0}

        def outer_counts_yields(request, state, runtime):
            response = yield request
            yield_count["value"] += 1
            return response

        def inner_retries(request, state, runtime):
            response = yield request
            if response.action == "raise":
                response = yield request  # Second attempt
            return response

        composed = _chain_model_call_handlers([outer_counts_yields, inner_retries])
        assert composed is not None

        gen = composed(create_test_request(), None, None)

        # First attempt - error
        request = next(gen)
        error = ModelResponse(action="raise", exception=ValueError("fail"))
        request = gen.send(error)

        # Second attempt - success
        success = ModelResponse(action="return", result=AIMessage(content="ok"))

        try:
            gen.send(success)
        except StopIteration as e:
            final_response = e.value

        assert yield_count["value"] == 1
        assert final_response.action == "return"
