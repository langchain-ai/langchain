from typing import Any


def get_headers(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError(
        "get_headers for LangChainTracerV1 is no longer supported. "
        "Please use LangChainTracer instead."
    )


def LangChainTracerV1(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError(
        "LangChainTracerV1 is no longer supported. Please use LangChainTracer instead."
    )
