from typing import Any


def comma_list(items: list[Any]) -> str:
    return ", ".join(str(item) for item in items)
