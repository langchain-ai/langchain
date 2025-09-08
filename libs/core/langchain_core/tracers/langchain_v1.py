"""This module is deprecated and will be removed in a future release.

Please use LangChainTracer instead.
"""

from typing import Any


def get_headers(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
    """Throw an error because this has been replaced by get_headers.

    Raises:
        RuntimeError: Always, because this function is deprecated.
    """
    msg = (
        "get_headers for LangChainTracerV1 is no longer supported. "
        "Please use LangChainTracer instead."
    )
    raise RuntimeError(msg)


def LangChainTracerV1(*args: Any, **kwargs: Any) -> Any:  # noqa: N802,ARG001
    """Throw an error because this has been replaced by ``LangChainTracer``.

    Raises:
        RuntimeError: Always, because this class is deprecated.
    """
    msg = (
        "LangChainTracerV1 is no longer supported. Please use LangChainTracer instead."
    )
    raise RuntimeError(msg)
