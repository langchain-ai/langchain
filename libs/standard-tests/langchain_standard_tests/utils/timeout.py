import inspect
import time
from typing import Any, Callable, TypeVar

import pytest

F = TypeVar("F", bound=Callable[..., Any])


def _timeout(*, seconds: float) -> Callable[[F], F]:
    """Decorator to measure the execution time of a test function and fail the test
    if it exceeds a specified maximum time.

    This function does **not** terminate the test function if it exceeds the maximum
    allowed time.

    Args:
        seconds: Maximum allowed time for the test function to execute, in seconds.

    Returns:
        Callable[[F], F]: A decorated function that measures execution time and
        enforces the maximum allowed time by failing the test if it is exceeded
        the allowed time.
    """

    def decorator(func: F) -> F:
        """Decorator function to wrap the test function.

        Args:
            func (F): The test function to be decorated.

        Returns:
            F: The wrapped test function.
        """

        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrapper for asynchronous test functions to measure execution time.

            Args:
                *args (Any): Positional arguments for the test function.
                **kwargs (Any): Keyword arguments for the test function.

            Returns:
                Any: The result of the test function.
            """
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            if duration > seconds:
                pytest.fail(
                    f"{func.__name__} exceeded the maximum allowed time of {seconds} "
                    f"seconds."
                )
            return result

        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrapper for synchronous test functions to measure execution time.

            Args:
                *args (Any): Positional arguments for the test function.
                **kwargs (Any): Keyword arguments for the test function.

            Returns:
                Any: The result of the test function.
            """
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            if duration > seconds:
                pytest.fail(
                    f"{func.__name__} exceeded the maximum allowed time of {seconds} "
                    f"seconds."
                )
            return result

        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore

    return decorator
