"""Tests for ContextThreadPoolExecutor."""
from contextvars import ContextVar

from langchain_core.runnables.config import ContextThreadPoolExecutor


def test_context_thread_pool_executor_map_with_generator() -> None:
    """Test that ContextThreadPoolExecutor.map() works with generators."""

    def values():
        yield from range(3)

    with ContextThreadPoolExecutor(max_workers=2) as executor:
        result = list(executor.map(lambda x: x * 2, values()))

    assert result == [0, 2, 4]


def test_context_thread_pool_executor_map_preserves_context() -> None:
    """Test that context variables are still passed to worker threads."""
    test_var: ContextVar[str] = ContextVar("test_var", default="not_set")
    test_var.set("caller_value")

    def get_var_and_multiply(x: int) -> str:
        return f"{test_var.get()}_{x}"

    def values():
        yield from range(2)

    with ContextThreadPoolExecutor(max_workers=1) as executor:
        result = list(executor.map(get_var_and_multiply, values()))

    assert result == ["caller_value_0", "caller_value_1"]


def test_context_thread_pool_executor_map_with_list() -> None:
    """Test that ContextThreadPoolExecutor.map() still works with lists."""
    with ContextThreadPoolExecutor(max_workers=2) as executor:
        result = list(executor.map(lambda x: x * 2, [0, 1, 2]))

    assert result == [0, 2, 4]


def test_context_thread_pool_executor_map_with_multiple_iterables() -> None:
    """Test shortest-iterable behavior is preserved with multiple iterables."""
    with ContextThreadPoolExecutor(max_workers=2) as executor:
        result = list(executor.map(lambda x, y: x + y, [1, 2, 3], [10, 20]))

    assert result == [11, 22]
