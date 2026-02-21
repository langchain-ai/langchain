"""Unit tests for TokenBudgetMiddleware."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from langchain.agents.factory import create_agent
from langchain.agents.middleware.token_budget import (
    TokenBudgetExceededError,
    TokenBudgetMiddleware,
    TokenBudgetState,
    _add_usage,
    _calculate_cost,
    _empty_usage,
)
from tests.unit_tests.agents.model import FakeToolCallingModel

# --- Initialization Validation ---


def test_middleware_initialization_validation() -> None:
    """Test that middleware initialization validates parameters correctly."""
    # Test that at least one limit must be specified
    with pytest.raises(ValueError, match="At least one limit must be specified"):
        TokenBudgetMiddleware()

    # Test valid initialization with token limits
    middleware = TokenBudgetMiddleware(
        thread_token_limit=100_000, run_token_limit=50_000
    )
    assert middleware.thread_token_limit == 100_000
    assert middleware.run_token_limit == 50_000
    assert middleware.exit_behavior == "continue"

    # Test valid initialization with only run limit
    middleware = TokenBudgetMiddleware(run_token_limit=50_000)
    assert middleware.run_token_limit == 50_000
    assert middleware.thread_token_limit is None

    # Test valid initialization with only thread limit
    middleware = TokenBudgetMiddleware(thread_token_limit=200_000)
    assert middleware.thread_token_limit == 200_000
    assert middleware.run_token_limit is None


def test_cost_limits_require_pricing() -> None:
    """Test that cost limits require per-token pricing to be specified."""
    with pytest.raises(
        ValueError,
        match="cost_per_input_token and cost_per_output_token must both be specified",
    ):
        TokenBudgetMiddleware(run_cost_limit=0.50)

    with pytest.raises(
        ValueError,
        match="cost_per_input_token and cost_per_output_token must both be specified",
    ):
        TokenBudgetMiddleware(
            run_cost_limit=0.50, cost_per_input_token=0.000003
        )

    # Valid cost-based initialization
    middleware = TokenBudgetMiddleware(
        run_cost_limit=0.50,
        cost_per_input_token=0.000003,
        cost_per_output_token=0.000015,
    )
    assert middleware.run_cost_limit == 0.50
    assert middleware.cost_per_input_token == 0.000003
    assert middleware.cost_per_output_token == 0.000015


def test_invalid_exit_behavior() -> None:
    """Test that invalid exit behavior raises ValueError."""
    with pytest.raises(ValueError, match="Invalid exit_behavior"):
        TokenBudgetMiddleware(
            run_token_limit=10_000,
            exit_behavior="invalid",  # type: ignore[arg-type]
        )


def test_run_limit_exceeds_thread_limit() -> None:
    """Test that run_token_limit cannot exceed thread_token_limit."""
    with pytest.raises(
        ValueError, match=r"run_token_limit .* cannot exceed thread_token_limit"
    ):
        TokenBudgetMiddleware(thread_token_limit=10_000, run_token_limit=50_000)

    # Equal limits should be valid
    middleware = TokenBudgetMiddleware(
        thread_token_limit=50_000, run_token_limit=50_000
    )
    assert middleware.thread_token_limit == 50_000
    assert middleware.run_token_limit == 50_000

    # Run limit less than thread limit should be valid
    middleware = TokenBudgetMiddleware(
        thread_token_limit=100_000, run_token_limit=50_000
    )
    assert middleware.thread_token_limit == 100_000
    assert middleware.run_token_limit == 50_000


def test_exit_behaviors() -> None:
    """Test that all valid exit behaviors are accepted."""
    for behavior in ["continue", "error", "end"]:
        middleware = TokenBudgetMiddleware(
            run_token_limit=10_000, exit_behavior=behavior  # type: ignore[arg-type]
        )
        assert middleware.exit_behavior == behavior


# --- Helper function tests ---


def test_empty_usage() -> None:
    """Test that _empty_usage returns zero-initialized dict."""
    usage = _empty_usage()
    assert usage == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def test_add_usage() -> None:
    """Test that _add_usage correctly adds token counts."""
    current = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
    result = _add_usage(current, input_tokens=200, output_tokens=100)
    assert result == {
        "input_tokens": 300,
        "output_tokens": 150,
        "total_tokens": 450,
    }

    # Starting from empty
    result = _add_usage(_empty_usage(), input_tokens=50, output_tokens=25)
    assert result == {
        "input_tokens": 50,
        "output_tokens": 25,
        "total_tokens": 75,
    }


def test_calculate_cost() -> None:
    """Test cost calculation from token usage."""
    usage = {"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500}
    cost = _calculate_cost(
        usage, cost_per_input_token=0.000003, cost_per_output_token=0.000015
    )
    expected = 1000 * 0.000003 + 500 * 0.000015
    assert abs(cost - expected) < 1e-10


# --- Token extraction tests ---


def test_token_extraction_with_usage_metadata() -> None:
    """Test that token usage is extracted from AIMessage.usage_metadata."""
    middleware = TokenBudgetMiddleware(run_token_limit=50_000)
    runtime = None

    state = TokenBudgetState(
        messages=[
            HumanMessage("What is 2+2?"),
            AIMessage(
                content="The answer is 4.",
                usage_metadata=UsageMetadata(
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15,
                ),
            ),
        ],
    )

    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert result["run_token_usage"]["input_tokens"] == 10
    assert result["run_token_usage"]["output_tokens"] == 5
    assert result["run_token_usage"]["total_tokens"] == 15
    assert result["thread_token_usage"]["input_tokens"] == 10
    assert result["thread_token_usage"]["output_tokens"] == 5
    assert result["thread_token_usage"]["total_tokens"] == 15


def test_token_extraction_fallback_no_usage_metadata() -> None:
    """Test fallback to approximate counting when usage_metadata is not available."""
    middleware = TokenBudgetMiddleware(run_token_limit=50_000)
    runtime = None

    state = TokenBudgetState(
        messages=[
            HumanMessage("What is 2+2?"),
            AIMessage(content="The answer is 4."),  # No usage_metadata
        ],
    )

    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    # Should have non-zero counts from approximate counting
    assert result["run_token_usage"]["total_tokens"] > 0
    assert result["thread_token_usage"]["total_tokens"] > 0


def test_no_ai_message_returns_none() -> None:
    """Test that after_model returns None if no AIMessage is found."""
    middleware = TokenBudgetMiddleware(run_token_limit=50_000)
    runtime = None

    state = TokenBudgetState(
        messages=[HumanMessage("Hello")],
    )

    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is None


def test_empty_messages_returns_none() -> None:
    """Test that after_model returns None if messages are empty."""
    middleware = TokenBudgetMiddleware(run_token_limit=50_000)
    runtime = None

    state = TokenBudgetState(messages=[])

    result = middleware.after_model(state, runtime)  # type: ignore[arg-type]
    assert result is None


# --- Budget enforcement tests ---


def test_run_token_limit_not_exceeded() -> None:
    """Test that before_model returns None when budget is not exceeded."""
    middleware = TokenBudgetMiddleware(run_token_limit=50_000)
    runtime = None

    state = TokenBudgetState(
        messages=[HumanMessage("Hi")],
        run_token_usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
    )

    result = middleware.before_model(state, runtime)  # type: ignore[arg-type]
    assert result is None


def test_run_token_limit_exceeded_end() -> None:
    """Test that 'end' behavior jumps to end when run budget is exceeded."""
    middleware = TokenBudgetMiddleware(run_token_limit=1000, exit_behavior="end")
    runtime = None

    state = TokenBudgetState(
        messages=[HumanMessage("Hi")],
        run_token_usage={
            "input_tokens": 800,
            "output_tokens": 300,
            "total_tokens": 1100,
        },
    )

    result = middleware.before_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert result["jump_to"] == "end"
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert "token budget exceeded" in result["messages"][0].content.lower()
    assert "run token limit" in result["messages"][0].content.lower()


def test_run_token_limit_exceeded_error() -> None:
    """Test that 'error' behavior raises TokenBudgetExceededError."""
    middleware = TokenBudgetMiddleware(run_token_limit=1000, exit_behavior="error")
    runtime = None

    state = TokenBudgetState(
        messages=[HumanMessage("Hi")],
        run_token_usage={
            "input_tokens": 800,
            "output_tokens": 300,
            "total_tokens": 1100,
        },
    )

    with pytest.raises(TokenBudgetExceededError) as exc_info:
        middleware.before_model(state, runtime)  # type: ignore[arg-type]

    error = exc_info.value
    assert error.run_token_limit == 1000
    assert error.run_usage["total_tokens"] == 1100


def test_run_token_limit_exceeded_continue() -> None:
    """Test that 'continue' behavior injects warning but doesn't jump to end."""
    middleware = TokenBudgetMiddleware(run_token_limit=1000, exit_behavior="continue")
    runtime = None

    state = TokenBudgetState(
        messages=[HumanMessage("Hi")],
        run_token_usage={
            "input_tokens": 800,
            "output_tokens": 300,
            "total_tokens": 1100,
        },
    )

    result = middleware.before_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    # Should NOT have jump_to
    assert "jump_to" not in result
    # Should have warning message
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert "budget exceeded" in result["messages"][0].content.lower()


def test_thread_token_limit_exceeded() -> None:
    """Test thread-level token limit enforcement."""
    middleware = TokenBudgetMiddleware(
        thread_token_limit=5000, exit_behavior="end"
    )
    runtime = None

    state = TokenBudgetState(
        messages=[HumanMessage("Hi")],
        thread_token_usage={
            "input_tokens": 3000,
            "output_tokens": 2500,
            "total_tokens": 5500,
        },
        run_token_usage=_empty_usage(),
    )

    result = middleware.before_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert result["jump_to"] == "end"
    assert "thread token limit" in result["messages"][0].content.lower()


def test_cost_limit_exceeded() -> None:
    """Test cost-based limit enforcement."""
    middleware = TokenBudgetMiddleware(
        run_cost_limit=0.01,  # $0.01
        cost_per_input_token=0.000003,  # $3/M
        cost_per_output_token=0.000015,  # $15/M
        exit_behavior="end",
    )
    runtime = None

    # Usage: 2000 input * $0.000003 + 1000 output * $0.000015 = $0.006 + $0.015 = $0.021
    state = TokenBudgetState(
        messages=[HumanMessage("Hi")],
        run_token_usage={
            "input_tokens": 2000,
            "output_tokens": 1000,
            "total_tokens": 3000,
        },
    )

    result = middleware.before_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert result["jump_to"] == "end"
    assert "cost limit" in result["messages"][0].content.lower()


def test_cost_limit_not_exceeded() -> None:
    """Test that cost limit is not triggered when within budget."""
    middleware = TokenBudgetMiddleware(
        run_cost_limit=1.00,  # $1.00
        cost_per_input_token=0.000003,
        cost_per_output_token=0.000015,
    )
    runtime = None

    # Usage: 100 input * $0.000003 + 50 output * $0.000015 = $0.00105
    state = TokenBudgetState(
        messages=[HumanMessage("Hi")],
        run_token_usage={
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        },
    )

    result = middleware.before_model(state, runtime)  # type: ignore[arg-type]
    assert result is None  # Budget not exceeded, should continue


def test_both_token_and_cost_limits() -> None:
    """Test that both token and cost limits can be set simultaneously."""
    middleware = TokenBudgetMiddleware(
        run_token_limit=10_000,
        run_cost_limit=0.50,
        cost_per_input_token=0.000003,
        cost_per_output_token=0.000015,
        exit_behavior="end",
    )
    runtime = None

    # Token limit exceeded, cost limit not
    state = TokenBudgetState(
        messages=[HumanMessage("Hi")],
        run_token_usage={
            "input_tokens": 8000,
            "output_tokens": 3000,
            "total_tokens": 11000,
        },
    )

    result = middleware.before_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert result["jump_to"] == "end"
    assert "token limit" in result["messages"][0].content.lower()


def test_limit_at_exact_boundary() -> None:
    """Test that limit is triggered at exact boundary (>=)."""
    middleware = TokenBudgetMiddleware(run_token_limit=1000, exit_behavior="end")
    runtime = None

    # At exactly the limit — should trigger
    state = TokenBudgetState(
        messages=[HumanMessage("Hi")],
        run_token_usage={
            "input_tokens": 600,
            "output_tokens": 400,
            "total_tokens": 1000,
        },
    )

    result = middleware.before_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert result["jump_to"] == "end"


def test_limit_just_below_boundary() -> None:
    """Test that limit is NOT triggered just below boundary."""
    middleware = TokenBudgetMiddleware(run_token_limit=1000, exit_behavior="end")
    runtime = None

    state = TokenBudgetState(
        messages=[HumanMessage("Hi")],
        run_token_usage={
            "input_tokens": 599,
            "output_tokens": 400,
            "total_tokens": 999,
        },
    )

    result = middleware.before_model(state, runtime)  # type: ignore[arg-type]
    assert result is None  # Budget not exceeded


# --- Cumulative tracking tests ---


def test_cumulative_token_tracking() -> None:
    """Test that token counts accumulate across multiple after_model calls."""
    middleware = TokenBudgetMiddleware(run_token_limit=100_000)
    runtime = None

    # First model call
    state1 = TokenBudgetState(
        messages=[
            HumanMessage("First question"),
            AIMessage(
                content="First answer",
                usage_metadata=UsageMetadata(
                    input_tokens=100, output_tokens=50, total_tokens=150
                ),
            ),
        ],
    )
    result1 = middleware.after_model(state1, runtime)  # type: ignore[arg-type]
    assert result1 is not None
    assert result1["run_token_usage"]["total_tokens"] == 150
    assert result1["thread_token_usage"]["total_tokens"] == 150

    # Second model call, with existing usage
    state2 = TokenBudgetState(
        messages=[
            HumanMessage("Second question"),
            AIMessage(
                content="Second answer",
                usage_metadata=UsageMetadata(
                    input_tokens=200, output_tokens=100, total_tokens=300
                ),
            ),
        ],
        thread_token_usage=result1["thread_token_usage"],
        run_token_usage=result1["run_token_usage"],
    )
    result2 = middleware.after_model(state2, runtime)  # type: ignore[arg-type]
    assert result2 is not None
    assert result2["run_token_usage"]["total_tokens"] == 450
    assert result2["thread_token_usage"]["total_tokens"] == 450
    assert result2["run_token_usage"]["input_tokens"] == 300
    assert result2["run_token_usage"]["output_tokens"] == 150


# --- Exception error messages ---


def test_exception_error_messages() -> None:
    """Test that error exception messages include expected information."""
    with pytest.raises(TokenBudgetExceededError) as exc_info:
        raise TokenBudgetExceededError(
            thread_usage={
                "input_tokens": 50000,
                "output_tokens": 60000,
                "total_tokens": 110000,
            },
            run_usage={
                "input_tokens": 30000,
                "output_tokens": 25000,
                "total_tokens": 55000,
            },
            thread_token_limit=100_000,
            run_token_limit=50_000,
        )
    msg = str(exc_info.value)
    assert "token budget exceeded" in msg.lower()
    assert "thread" in msg.lower()
    assert "run" in msg.lower()


# --- Integration test ---


def test_integration_with_create_agent() -> None:
    """Test end-to-end integration with create_agent and FakeToolCallingModel."""

    @tool
    def search(query: str) -> str:
        """Search for information."""
        return f"Results for {query}"

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test1"}, id="1")],
            [ToolCall(name="search", args={"query": "test2"}, id="2")],
            [ToolCall(name="search", args={"query": "test3"}, id="3")],
            [],
        ]
    )

    # Set a very low token limit to trigger budget enforcement
    # Note: FakeToolCallingModel doesn't set usage_metadata, so the middleware
    # will use the fallback approximate counting
    budget = TokenBudgetMiddleware(run_token_limit=50, exit_behavior="end")
    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[budget],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Run all searches")]},
        {"configurable": {"thread_id": "test_integration"}},
    )

    # Agent should eventually stop due to token budget
    messages = result["messages"]
    assert len(messages) > 0

    ai_limit_messages = [
        msg
        for msg in messages
        if isinstance(msg, AIMessage)
        and isinstance(msg.content, str)
        and "budget exceeded" in msg.content.lower()
    ]

    # With a very low token limit and approximate counting, budget should be hit
    assert len(ai_limit_messages) > 0, (
        "Should have at least one AI message about budget exceeded"
    )


def test_integration_no_limit_exceeded() -> None:
    """Test integration where the budget is high enough to not be exceeded."""

    @tool
    def search(query: str) -> str:
        """Search for information."""
        return f"Results for {query}"

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test1"}, id="1")],
            [],
        ]
    )

    # Set a very high token limit
    budget = TokenBudgetMiddleware(run_token_limit=1_000_000)
    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[budget],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Search for test1")]},
        {"configurable": {"thread_id": "test_no_exceed"}},
    )

    messages = result["messages"]

    # Should have completed without budget-related messages
    budget_messages = [
        msg
        for msg in messages
        if isinstance(msg, AIMessage)
        and isinstance(msg.content, str)
        and "budget exceeded" in msg.content.lower()
    ]
    assert len(budget_messages) == 0, (
        "Should not have any budget exceeded messages"
    )

    # Should have successful tool execution
    tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
    assert len(tool_messages) > 0, "Should have tool execution results"


async def test_async_methods_delegate_to_sync() -> None:
    """Test that async methods delegate to their sync counterparts."""
    middleware = TokenBudgetMiddleware(run_token_limit=1000, exit_behavior="end")
    runtime = None

    # Test abefore_model delegates to before_model
    state = TokenBudgetState(
        messages=[HumanMessage("Hi")],
        run_token_usage={
            "input_tokens": 800,
            "output_tokens": 300,
            "total_tokens": 1100,
        },
    )

    result = await middleware.abefore_model(state, runtime)  # type: ignore[arg-type]
    assert result is not None
    assert result["jump_to"] == "end"

    # Test aafter_model delegates to after_model
    state2 = TokenBudgetState(
        messages=[
            HumanMessage("Question"),
            AIMessage(
                content="Answer",
                usage_metadata=UsageMetadata(
                    input_tokens=10, output_tokens=5, total_tokens=15
                ),
            ),
        ],
    )

    result2 = await middleware.aafter_model(state2, runtime)  # type: ignore[arg-type]
    assert result2 is not None
    assert result2["run_token_usage"]["total_tokens"] == 15
