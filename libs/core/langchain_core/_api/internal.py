import inspect


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
        # Directly access the module name from the frame's global variables
        module_globals = frame.f_globals
        caller_module_name = module_globals.get("__name__", "")
        return caller_module_name.startswith("langchain")
    finally:
        del frame
