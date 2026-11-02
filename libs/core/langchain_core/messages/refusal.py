"""Detection of model refusals in messages."""

from typing import cast

from langchain_core.messages.ai import AIMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.messages.content import RefusalContentBlock


def get_refusal(message: BaseMessage) -> RefusalContentBlock | None:
    """Return the refusal block of `message`, if the model refused.

    Reads `message.content_blocks`, so it works the same for every provider whose
    refusals are translated to a `RefusalContentBlock`. It also recognizes OpenAI
    refusals stored as `non_standard` blocks by earlier versions.

    !!! version-added "Added in `langchain-core` 1.7.5"

    Args:
        message: The message to check.

    Returns:
        The first refusal block, or `None` if `message` is not an AI message or
            has no refusal.

    Example:
        ```python
        from langchain_core.exceptions import ModelRefusalError
        from langchain_core.messages import get_refusal

        response = model.invoke("...")
        if refusal := get_refusal(response):
            raise ModelRefusalError.from_refusal(refusal, raw_message=response)
        ```
    """
    if not isinstance(message, AIMessage):
        return None
    for block in message.content_blocks:
        if block["type"] == "refusal":
            return block
        if block["type"] == "non_standard" and block["value"].get("type") == "refusal":
            return cast("RefusalContentBlock", block["value"])
    return None
