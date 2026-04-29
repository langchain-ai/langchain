"""Additional unit tests for `ModelCallLimitMiddleware`.

These tests complement `test_model_call_limit.py` by covering the async hooks,
the `_build_limit_exceeded_message` helper, `after_model` counting logic, and
boundary / combined-limit scenarios.
"""

import pytest
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from langchain.agents.middleware.model_call_limit import (
    ModelCallLimitExceededError,
    ModelCallLimitMiddleware,
    ModelCallLimitState,
    _build_limit_exceeded_message,
)


class TestBuildLimitExceededMessage:
    """Direct tests for the `_build_limit_exceeded_message` helper."""

    def test_thread_only_exceeded(self) -> None:
        """Only the thread fragment is rendered when run limit is unset."""
        msg = _build_limit_exceeded_message(
            thread_count=2, run_count=0, thread_limit=2, run_limit=None
        )
        assert "thread limit (2/2)" in msg
        assert "run limit" not in msg

    def test_run_only_exceeded(self) -> None:
        """Only the run fragment is rendered when thread limit is unset."""
        msg = _build_limit_exceeded_message(
            thread_count=0, run_count=3, thread_limit=None, run_limit=3
        )
        assert "run limit (3/3)" in msg
        assert "thread limit" not in msg

    def test_both_exceeded(self) -> None:
        """Both fragments are joined verbatim when both limits are exceeded.

        This test asserts the full expected string to guard against silent
        formatting regressions in the helper. Other tests in this class use
        substring checks to remain tolerant of trivial wording tweaks.
        """
        msg = _build_limit_exceeded_message(
            thread_count=2, run_count=1, thread_limit=2, run_limit=1
        )
        assert msg == "Model call limits exceeded: thread limit (2/2), run limit (1/1)"

    def test_thread_count_above_limit(self) -> None:
        """The helper reports the actual count even when it exceeds the limit."""
        msg = _build_limit_exceeded_message(
            thread_count=5, run_count=0, thread_limit=2, run_limit=None
        )
        assert "thread limit (5/2)" in msg

    def test_neither_exceeded_documents_current_behavior(self) -> None:
        """Characterize current behavior when neither limit is exceeded.

        The helper unconditionally returns the `Model call limits exceeded:`
        prefix even when the exceeded list is empty. Callers in this module
        only invoke the helper after confirming a breach, so this branch is
        not reached in production. If the helper is ever updated to short-
        circuit on an empty list, this test should be updated alongside it.
        """
        msg = _build_limit_exceeded_message(
            thread_count=0, run_count=0, thread_limit=5, run_limit=5
        )
        assert msg == "Model call limits exceeded: "


class TestAsyncBeforeModel:
    """Tests for the async `abefore_model` hook."""

    @pytest.mark.asyncio
    async def test_abefore_model_returns_none_when_under_limit(self) -> None:
        """Async hook returns `None` when no limit is breached."""
        middleware = ModelCallLimitMiddleware(thread_limit=5, run_limit=5)
        state = ModelCallLimitState(messages=[], thread_model_call_count=0, run_model_call_count=0)
        result = await middleware.abefore_model(state, Runtime())
        assert result is None

    @pytest.mark.asyncio
    async def test_abefore_model_returns_jump_to_end_on_thread_limit(self) -> None:
        """Async hook emits a `jump_to=end` payload when the thread limit is hit."""
        middleware = ModelCallLimitMiddleware(thread_limit=2, run_limit=5)
        state = ModelCallLimitState(messages=[], thread_model_call_count=2, run_model_call_count=0)
        result = await middleware.abefore_model(state, Runtime())
        assert result is not None
        assert result["jump_to"] == "end"
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert "thread limit (2/2)" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_abefore_model_returns_jump_to_end_on_run_limit(self) -> None:
        """Async hook emits a `jump_to=end` payload when the run limit is hit."""
        middleware = ModelCallLimitMiddleware(thread_limit=5, run_limit=2)
        state = ModelCallLimitState(messages=[], thread_model_call_count=0, run_model_call_count=2)
        result = await middleware.abefore_model(state, Runtime())
        assert result is not None
        assert result["jump_to"] == "end"
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert "run limit (2/2)" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_abefore_model_raises_with_error_behavior(self) -> None:
        """Async hook raises `ModelCallLimitExceededError` when configured."""
        middleware = ModelCallLimitMiddleware(thread_limit=2, run_limit=1, exit_behavior="error")
        state = ModelCallLimitState(messages=[], thread_model_call_count=2, run_model_call_count=0)
        with pytest.raises(ModelCallLimitExceededError) as exc_info:
            await middleware.abefore_model(state, Runtime())
        assert "thread limit (2/2)" in str(exc_info.value)


class TestAsyncAfterModel:
    """Tests for the async `aafter_model` hook."""

    @pytest.mark.asyncio
    async def test_aafter_model_increments_counts(self) -> None:
        """Async after-hook increments both counts by one."""
        middleware = ModelCallLimitMiddleware(thread_limit=10)
        state = ModelCallLimitState(messages=[], thread_model_call_count=2, run_model_call_count=5)
        result = await middleware.aafter_model(state, Runtime())
        assert result == {"thread_model_call_count": 3, "run_model_call_count": 6}

    @pytest.mark.asyncio
    async def test_aafter_model_starts_from_zero_when_keys_missing(self) -> None:
        """Async after-hook treats missing count keys as zero."""
        middleware = ModelCallLimitMiddleware(thread_limit=10)
        state = ModelCallLimitState(messages=[])
        result = await middleware.aafter_model(state, Runtime())
        assert result == {"thread_model_call_count": 1, "run_model_call_count": 1}


class TestAfterModelCounting:
    """Tests for the synchronous `after_model` counting logic."""

    def test_after_model_increments_from_zero(self) -> None:
        """Counts default to zero when keys are absent and increment to one."""
        middleware = ModelCallLimitMiddleware(thread_limit=10)
        state = ModelCallLimitState(messages=[])
        result = middleware.after_model(state, Runtime())
        assert result == {"thread_model_call_count": 1, "run_model_call_count": 1}

    def test_after_model_returns_dict_with_expected_keys(self) -> None:
        """Returned update dict exposes exactly the two count keys."""
        middleware = ModelCallLimitMiddleware(thread_limit=10)
        state = ModelCallLimitState(messages=[], thread_model_call_count=0, run_model_call_count=0)
        result = middleware.after_model(state, Runtime())
        assert result is not None
        assert set(result.keys()) == {"thread_model_call_count", "run_model_call_count"}

    def test_after_model_increments_independently(self) -> None:
        """Each count is incremented from its own current value."""
        middleware = ModelCallLimitMiddleware(thread_limit=20)
        state = ModelCallLimitState(messages=[], thread_model_call_count=10, run_model_call_count=2)
        result = middleware.after_model(state, Runtime())
        assert result == {"thread_model_call_count": 11, "run_model_call_count": 3}

    def test_after_model_does_not_mutate_input_state(self) -> None:
        """The hook returns an update dict without mutating the input state."""
        middleware = ModelCallLimitMiddleware(thread_limit=10)
        state = ModelCallLimitState(messages=[], thread_model_call_count=4, run_model_call_count=1)
        middleware.after_model(state, Runtime())
        assert state["thread_model_call_count"] == 4
        assert state["run_model_call_count"] == 1


class TestBoundaryConditions:
    """Boundary-value tests around the `>=` comparison in `before_model`."""

    def test_count_exactly_at_thread_limit_triggers_exit(self) -> None:
        """`thread_count == thread_limit` should breach the limit."""
        middleware = ModelCallLimitMiddleware(thread_limit=3)
        state = ModelCallLimitState(messages=[], thread_model_call_count=3, run_model_call_count=0)
        result = middleware.before_model(state, Runtime())
        assert result is not None
        assert result["jump_to"] == "end"

    def test_count_one_below_thread_limit_does_not_trigger(self) -> None:
        """`thread_count == thread_limit - 1` should not breach the limit."""
        middleware = ModelCallLimitMiddleware(thread_limit=3)
        state = ModelCallLimitState(messages=[], thread_model_call_count=2, run_model_call_count=0)
        assert middleware.before_model(state, Runtime()) is None

    def test_count_exactly_at_run_limit_triggers_exit(self) -> None:
        """`run_count == run_limit` should breach the limit."""
        middleware = ModelCallLimitMiddleware(run_limit=2)
        state = ModelCallLimitState(messages=[], thread_model_call_count=0, run_model_call_count=2)
        result = middleware.before_model(state, Runtime())
        assert result is not None
        assert result["jump_to"] == "end"

    def test_zero_count_with_limit_of_one_does_not_trigger(self) -> None:
        """A fresh state with a small limit must still allow the first call."""
        middleware = ModelCallLimitMiddleware(thread_limit=1, run_limit=1)
        state = ModelCallLimitState(messages=[], thread_model_call_count=0, run_model_call_count=0)
        assert middleware.before_model(state, Runtime()) is None


class TestCombinedScenarios:
    """Tests for combined thread + run limit interactions."""

    def test_thread_exceeded_run_not_exceeded(self) -> None:
        """Only the thread fragment appears when only the thread limit is hit."""
        middleware = ModelCallLimitMiddleware(thread_limit=2, run_limit=5)
        state = ModelCallLimitState(messages=[], thread_model_call_count=2, run_model_call_count=1)
        result = middleware.before_model(state, Runtime())
        assert result is not None
        assert isinstance(result["messages"][0], AIMessage)
        content = result["messages"][0].content
        assert "thread limit (2/2)" in content
        assert "run limit" not in content

    def test_run_exceeded_thread_not_exceeded(self) -> None:
        """Only the run fragment appears when only the run limit is hit."""
        middleware = ModelCallLimitMiddleware(thread_limit=5, run_limit=2)
        state = ModelCallLimitState(messages=[], thread_model_call_count=1, run_model_call_count=2)
        result = middleware.before_model(state, Runtime())
        assert result is not None
        assert isinstance(result["messages"][0], AIMessage)
        content = result["messages"][0].content
        assert "run limit (2/2)" in content
        assert "thread limit" not in content

    def test_both_exceeded_message_lists_thread_before_run(self) -> None:
        """When both limits are exceeded, thread is reported before run."""
        middleware = ModelCallLimitMiddleware(thread_limit=2, run_limit=1)
        state = ModelCallLimitState(messages=[], thread_model_call_count=2, run_model_call_count=1)
        result = middleware.before_model(state, Runtime())
        assert result is not None
        assert isinstance(result["messages"][0], AIMessage)
        content = result["messages"][0].content
        assert content.index("thread limit") < content.index("run limit")

    def test_both_exceeded_with_error_behavior_attaches_all_counts(self) -> None:
        """The raised error preserves both counts and both limits as attributes."""
        middleware = ModelCallLimitMiddleware(thread_limit=2, run_limit=1, exit_behavior="error")
        state = ModelCallLimitState(messages=[], thread_model_call_count=2, run_model_call_count=1)
        with pytest.raises(ModelCallLimitExceededError) as exc_info:
            middleware.before_model(state, Runtime())
        err = exc_info.value
        assert err.thread_count == 2
        assert err.run_count == 1
        assert err.thread_limit == 2
        assert err.run_limit == 1


class TestEdgeCases:
    """Edge cases around partial limits and missing state keys."""

    def test_only_thread_limit_set_run_count_irrelevant(self) -> None:
        """A large run count cannot trigger an exit when `run_limit` is `None`."""
        middleware = ModelCallLimitMiddleware(thread_limit=5)
        state = ModelCallLimitState(
            messages=[], thread_model_call_count=0, run_model_call_count=999
        )
        assert middleware.before_model(state, Runtime()) is None

    def test_only_run_limit_set_thread_count_irrelevant(self) -> None:
        """A large thread count cannot trigger an exit when `thread_limit` is `None`."""
        middleware = ModelCallLimitMiddleware(run_limit=5)
        state = ModelCallLimitState(
            messages=[], thread_model_call_count=999, run_model_call_count=0
        )
        assert middleware.before_model(state, Runtime()) is None

    def test_missing_count_keys_treated_as_zero(self) -> None:
        """Absent count keys default to zero rather than raising."""
        middleware = ModelCallLimitMiddleware(thread_limit=1, run_limit=1)
        state = ModelCallLimitState(messages=[])
        assert middleware.before_model(state, Runtime()) is None
