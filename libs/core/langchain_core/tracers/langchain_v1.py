from typing import Any


def get_headers(*args: Any, **kwargs: Any) -> Any:
    """Throw an error because this has been replaced by get_headers."""
    raise RuntimeError(
        "get_headers for LangChainTracerV1 is no longer supported. "
        "Please use LangChainTracer instead."
    )


def LangChainTracerV1(*args: Any, **kwargs: Any) -> Any:  # noqa: N802
    """Throw an error because this has been replaced by LangChainTracer."""
    raise RuntimeError(
        "LangChainTracerV1 is no longer supported. Please use LangChainTracer instead."
    )
