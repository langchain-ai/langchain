def safe_repr(obj, seen=None):
    """A repr implementation that accounts for recursive object references."""

    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        raise RecursionError("Found a recursive reference")

    seen.add(obj_id)
    result = []

    if isinstance(obj, dict):
        result.append("{")
        for key, value in obj.items():
            result.append(f"{safe_repr(key, seen=seen)}: {safe_repr(value, seen=seen)}, ")
        result.append("}")
    elif isinstance(obj, list):
        result.append("[")
        result.extend(f"{safe_repr(item, seen=seen)}, " for item in obj)
        result.append("]")
    elif isinstance(obj, tuple):
        result.append("(")
        result.extend(f"{safe_repr(item, seen=seen)}, " for item in obj)
        result.append(")")
    elif isinstance(obj, set):
        result.append("{")
        result.extend(f"{safe_repr(item, seen=seen)}, " for item in obj)
        result.append("}")
    else:
        # Handle general class instances
        if hasattr(obj, "__dict__"):
            result.append(f"<{obj.__class__.__name__}: ")
            for key, value in obj.__dict__.items():
                result.append(f"{key}={safe_repr(value, seen=seen)}, ")
            result.append(">")
        else:
            # Fallback for objects without __dict__ or non-container types
            result.append(repr(obj))

    seen.remove(obj_id)
    return ''.join(result)
