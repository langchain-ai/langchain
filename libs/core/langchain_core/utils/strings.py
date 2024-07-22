from typing import Any, List


def stringify_value(val: Any) -> str:
    """Stringify a value.

    Args:
        val: The value to stringify.

    Returns:
        str: The stringified value.
    """
    if isinstance(val, str):
        return val
    elif isinstance(val, dict):
        return "\n" + stringify_dict(val)
    elif isinstance(val, list):
        return "\n".join(stringify_value(v) for v in val)
    else:
        return str(val)


def stringify_dict(data: dict) -> str:
    """Stringify a dictionary.

    Args:
        data: The dictionary to stringify.

    Returns:
        str: The stringified dictionary.
    """
    text = ""
    for key, value in data.items():
        text += key + ": " + stringify_value(value) + "\n"
    return text


def comma_list(items: List[Any]) -> str:
    """Convert a list to a comma-separated string.

    Args:
        items: The list to convert.

    Returns:
        str: The comma-separated string.
    """
    return ", ".join(str(item) for item in items)
