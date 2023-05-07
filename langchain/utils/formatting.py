from typing import Any


def comma_list(items: list[Any]) -> str:
    return ", ".join(str(item) for item in items)


def bullet_list(items: list[Any]) -> str:
    return "\n".join(f"- {str(item)}" for item in items)


def summarized_items(items: list[str], chars=30) -> str:
    return "\n\n".join(
        f"{str(item)[:chars]}..." if len(item) > chars else item for item in items
    )
