import inspect

_INTERNAL_PREFIXES = [
    "langchain",  # For example langchain-core, langchain, langchain-anthropic
    "langserve",
    "langgraph",
]


def is_caller_internal(depth: int = 2) -> bool:
    """Return whether the caller at `depth` of this function is internal."""
    try:
        frame = inspect.currentframe()
    except AttributeError:
        return False
    if frame is None:
        return False
    try:
        for _ in range(depth):
            frame = frame.f_back
            if frame is None:
                return False
        caller_module = inspect.getmodule(frame)
        if caller_module is None:
            return False
        caller_module_name = caller_module.__name__

        for prefix in _INTERNAL_PREFIXES:
            if caller_module_name.startswith(prefix):
                return True
    finally:
        del frame

    return False
