import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.runtime import Runtime

from langchain.agents.middleware.token_usage_tracking import (
    TokenBudgetExceededError,
    TokenUsageState,
    TokenUsageTrackingMiddleware,
    _build_budget_exceeded_message,
    _extract_usage,
)


@tool
def simple_tool(value: str) -> str:
    """A simple tool."""
    return value


def test_extract_usage_from_ai_message() -> None:
    """Test extracting usage metadata from an AIMessage."""
    ai_msg = AIMessage(
        content="hello",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    state = TokenUsageState(messages=[HumanMessage(content="hi"), ai_msg])
    assert _extract_usage(state) == (10, 5, 15)


def test_extract_usage_no_metadata() -> None:
    """Test that missing usage_metadata returns zeros."""
    ai_msg = AIMessage(content="hello")
    state = TokenUsageState(messages=[HumanMessage(content="hi"), ai_msg])
    assert _extract_usage(state) == (0, 0, 0)


def test_extract_usage_empty_messages() -> None:
    """Test extraction from empty message list."""
    state = TokenUsageState(messages=[])
    assert _extract_usage(state) == (0, 0, 0)


def test_extract_usage_picks_latest_ai_message() -> None:
    """Test that the most recent AIMessage with metadata is used."""
    old_msg = AIMessage(
        content="old",
        usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    )
    new_msg = AIMessage(
        content="new",
        usage_metadata={"input_tokens": 50, "output_tokens": 25, "total_tokens": 75},
    )
    state = TokenUsageState(messages=[old_msg, HumanMessage(content="q"), new_msg])
    assert _extract_usage(state) == (50, 25, 75)


def test_build_budget_exceeded_message_thread() -> None:
    """Test budget exceeded message for thread budget."""
    msg = _build_budget_exceeded_message(
        thread_total=1000, run_total=500, thread_budget=1000, run_budget=None
    )
    assert "thread budget (1000/1000 tokens)" in msg


def test_build_budget_exceeded_message_run() -> None:
    """Test budget exceeded message for run budget."""
    msg = _build_budget_exceeded_message(
        thread_total=500, run_total=1000, thread_budget=None, run_budget=1000
    )
    assert "run budget (1000/1000 tokens)" in msg


def test_build_budget_exceeded_message_both() -> None:
    """Test budget exceeded message when both budgets exceeded."""
    msg = _build_budget_exceeded_message(
        thread_total=2000, run_total=1000, thread_budget=2000, run_budget=1000
    )
    assert "thread budget" in msg
    assert "run budget" in msg


def test_middleware_no_budgets_tracking_only() -> None:
    """Test middleware with no budgets set (tracking only mode)."""
    middleware = TokenUsageTrackingMiddleware()
    runtime = Runtime()

    # before_model should always return None with no budgets
    state = TokenUsageState(
        messages=[], thread_total_tokens=999999, run_total_tokens=999999
    )
    assert middleware.before_model(state, runtime) is None


def test_middleware_after_model_accumulates_tokens() -> None:
    """Test that after_model correctly accumulates token counts."""
    middleware = TokenUsageTrackingMiddleware()
    runtime = Runtime()

    ai_msg = AIMessage(
        content="response",
        usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
    )
    state = TokenUsageState(
        messages=[HumanMessage(content="hi"), ai_msg],
        thread_input_tokens=200,
        thread_output_tokens=100,
        thread_total_tokens=300,
        run_input_tokens=0,
        run_output_tokens=0,
        run_total_tokens=0,
    )

    result = middleware.after_model(state, runtime)
    assert result is not None
    assert result["thread_input_tokens"] == 300  # 200 + 100
    assert result["thread_output_tokens"] == 150  # 100 + 50
    assert result["thread_total_tokens"] == 450  # 300 + 150
    assert result["run_input_tokens"] == 100
    assert result["run_output_tokens"] == 50
    assert result["run_total_tokens"] == 150


def test_middleware_after_model_no_usage_returns_none() -> None:
    """Test that after_model returns None when no usage metadata found."""
    middleware = TokenUsageTrackingMiddleware()
    runtime = Runtime()

    state = TokenUsageState(messages=[AIMessage(content="no usage")])
    assert middleware.after_model(state, runtime) is None


def test_middleware_after_model_defaults_to_zero() -> None:
    """Test that after_model initializes counts from zero when state is empty."""
    middleware = TokenUsageTrackingMiddleware()
    runtime = Runtime()

    ai_msg = AIMessage(
        content="first",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    state = TokenUsageState(messages=[ai_msg])

    result = middleware.after_model(state, runtime)
    assert result is not None
    assert result["thread_input_tokens"] == 10
    assert result["thread_output_tokens"] == 5
    assert result["thread_total_tokens"] == 15
    assert result["run_input_tokens"] == 10
    assert result["run_output_tokens"] == 5
    assert result["run_total_tokens"] == 15


def test_middleware_before_model_thread_budget_exceeded_end() -> None:
    """Test before_model ends agent when thread budget is exceeded."""
    middleware = TokenUsageTrackingMiddleware(thread_budget=500)
    runtime = Runtime()

    state = TokenUsageState(messages=[], thread_total_tokens=500)
    result = middleware.before_model(state, runtime)
    assert result is not None
    assert result["jump_to"] == "end"
    assert len(result["messages"]) == 1
    assert "thread budget (500/500 tokens)" in result["messages"][0].content


def test_middleware_before_model_run_budget_exceeded_end() -> None:
    """Test before_model ends agent when run budget is exceeded."""
    middleware = TokenUsageTrackingMiddleware(run_budget=1000)
    runtime = Runtime()

    state = TokenUsageState(messages=[], run_total_tokens=1000)
    result = middleware.before_model(state, runtime)
    assert result is not None
    assert result["jump_to"] == "end"
    assert "run budget (1000/1000 tokens)" in result["messages"][0].content


def test_middleware_before_model_under_budget() -> None:
    """Test before_model returns None when under budget."""
    middleware = TokenUsageTrackingMiddleware(thread_budget=1000, run_budget=500)
    runtime = Runtime()

    state = TokenUsageState(
        messages=[], thread_total_tokens=100, run_total_tokens=50
    )
    assert middleware.before_model(state, runtime) is None


def test_middleware_before_model_error_behavior() -> None:
    """Test before_model raises TokenBudgetExceededError with error behavior."""
    middleware = TokenUsageTrackingMiddleware(
        thread_budget=500, exit_behavior="error"
    )
    runtime = Runtime()

    state = TokenUsageState(messages=[], thread_total_tokens=500)
    with pytest.raises(TokenBudgetExceededError) as exc_info:
        middleware.before_model(state, runtime)

    assert exc_info.value.thread_total == 500
    assert exc_info.value.thread_budget == 500
    assert "thread budget" in str(exc_info.value)


def test_middleware_before_model_run_budget_error() -> None:
    """Test before_model raises error when run budget is exceeded."""
    middleware = TokenUsageTrackingMiddleware(
        run_budget=200, exit_behavior="error"
    )
    runtime = Runtime()

    state = TokenUsageState(messages=[], run_total_tokens=200)
    with pytest.raises(TokenBudgetExceededError) as exc_info:
        middleware.before_model(state, runtime)

    assert exc_info.value.run_total == 200
    assert exc_info.value.run_budget == 200


def test_invalid_exit_behavior() -> None:
    """Test that invalid exit_behavior raises ValueError."""
    with pytest.raises(ValueError, match="Invalid exit_behavior"):
        TokenUsageTrackingMiddleware(thread_budget=100, exit_behavior="invalid")  # type: ignore[arg-type]


def test_token_budget_exceeded_error_attributes() -> None:
    """Test TokenBudgetExceededError stores all attributes."""
    err = TokenBudgetExceededError(
        thread_total=1000,
        run_total=500,
        thread_budget=800,
        run_budget=400,
    )
    assert err.thread_total == 1000
    assert err.run_total == 500
    assert err.thread_budget == 800
    assert err.run_budget == 400
    assert "thread budget" in str(err)
    assert "run budget" in str(err)


@pytest.mark.asyncio
async def test_abefore_model_delegates_to_sync() -> None:
    """Test that abefore_model delegates to before_model."""
    middleware = TokenUsageTrackingMiddleware(run_budget=100)
    runtime = Runtime()

    state = TokenUsageState(messages=[], run_total_tokens=100)
    result = await middleware.abefore_model(state, runtime)
    assert result is not None
    assert result["jump_to"] == "end"


@pytest.mark.asyncio
async def test_aafter_model_delegates_to_sync() -> None:
    """Test that aafter_model delegates to after_model."""
    middleware = TokenUsageTrackingMiddleware()
    runtime = Runtime()

    ai_msg = AIMessage(
        content="resp",
        usage_metadata={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
    )
    state = TokenUsageState(messages=[ai_msg])
    result = await middleware.aafter_model(state, runtime)
    assert result is not None
    assert result["run_total_tokens"] == 30


def test_extract_usage_partial_metadata() -> None:
    """Test extraction when usage_metadata has missing keys."""
    ai_msg = AIMessage(
        content="partial",
        usage_metadata={"input_tokens": 5, "output_tokens": 0, "total_tokens": 5},
    )
    state = TokenUsageState(messages=[ai_msg])
    assert _extract_usage(state) == (5, 0, 5)
