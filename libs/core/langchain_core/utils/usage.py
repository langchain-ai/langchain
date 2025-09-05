"""Usage utilities."""

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from langchain_core.messages.ai import UsageMetadata


def _dict_int_op(
    left: dict,
    right: dict,
    op: Callable[[int, int], int],
    *,
    default: int = 0,
    depth: int = 0,
    max_depth: int = 100,
) -> dict:
    if depth >= max_depth:
        msg = f"{max_depth=} exceeded, unable to combine dicts."
        raise ValueError(msg)
    combined: dict = {}
    for k in set(left).union(right):
        if isinstance(left.get(k, default), int) and isinstance(
            right.get(k, default), int
        ):
            combined[k] = op(left.get(k, default), right.get(k, default))
        elif isinstance(left.get(k, {}), dict) and isinstance(right.get(k, {}), dict):
            combined[k] = _dict_int_op(
                left.get(k, {}),
                right.get(k, {}),
                op,
                default=default,
                depth=depth + 1,
                max_depth=max_depth,
            )
        else:
            types = [type(d[k]) for d in (left, right) if k in d]
            msg = (
                f"Unknown value types: {types}. Only dict and int values are supported."
            )
            raise ValueError(msg)  # noqa: TRY004
    return combined


def get_billable_input_tokens(usage_metadata: "UsageMetadata") -> int:
    """Calculate billable input tokens excluding cached tokens.

    When using prompt caching (e.g., with Anthropic models), the ``input_tokens``
    field on ``UsageMetadata`` represents the total tokens processed (cached +
    non-cached), but you're only charged for non-cached tokens. This function calculates
    the actual billable input tokens.

    Example:
        .. code-block:: python

            from langchain_anthropic import ChatAnthropic
            from langchain_core.utils.usage import get_billable_input_tokens

            model = ChatAnthropic(model="claude-3-sonnet-20240229")
            response = model.invoke([{"role": "user", "content": "Hello!"}])

            # Calculate billable tokens
            billable = get_billable_input_tokens(response.usage_metadata)

    """
    total_input = usage_metadata["input_tokens"]
    details = usage_metadata.get("input_token_details", {})
    cache_read = details.get("cache_read", 0)
    cache_creation = details.get("cache_creation", 0)
    return total_input - cache_read - cache_creation
