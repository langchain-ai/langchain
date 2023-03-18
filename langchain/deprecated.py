"""Take from stack overflow.

https://stackoverflow.com/questions/2536307/decorators-in-the-python-standard-lib-deprecated-specifically.
"""
import functools
import warnings
from typing import Any, Callable


def deprecated(reason: str) -> Callable:
    """Decorator for deprecation warnings."""

    def decorator(func: Any) -> Any:
        @functools.wraps(func)
        def new_func1(*args: Any, **kwargs: Any) -> Any:
            warnings.simplefilter("always", DeprecationWarning)
            warnings.warn(
                reason,
                category=DeprecationWarning,
                stacklevel=2,
            )
            warnings.simplefilter("default", DeprecationWarning)
            return func(*args, **kwargs)

        return new_func1

    return decorator
