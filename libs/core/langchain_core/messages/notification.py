"""Metadata for messages the agent harness adds to a conversation."""

from typing import Any

from typing_extensions import NotRequired, TypedDict

from langchain_core.messages.base import BaseMessage
from langchain_core.messages.human import HumanMessage


class NotificationMetadata(TypedDict):
    """Which feature added a message, and that feature's fields.

    !!! version-added "Added in `langchain-core` 1.7.2"
    """

    source: str
    """The feature that added the message, e.g. `'async_subagent'`."""

    data: NotRequired[dict[str, Any]]
    """The feature's fields, documented by that feature."""


def add_notification_metadata(
    message: HumanMessage, *, source: str, data: dict[str, Any] | None = None
) -> HumanMessage:
    """Return a copy of `message` marked as added by the agent harness.

    Use it for a human message the user didn't type, such as a background task
    update or a conversation summary. It sets `additional_kwargs["lc_source"]`
    and, when `data` is given, `additional_kwargs["lc_data"]`. Any earlier
    `lc_source` and `lc_data` are removed first. LangChain's chat model
    integrations don't send these keys to the model.

    It doesn't change the content. Write the content the model reads yourself.

    Anyone who can send input to a thread can set these keys. Don't use them to
    decide whether to trust a message.

    `merge_message_runs` and chunk addition join a notification with adjacent
    human messages; the merged message keeps the metadata and concatenates
    `lc_source`.

    !!! version-added "Added in `langchain-core` 1.7.2"

    Args:
        message: The message to mark.
        source: The feature that added the message.
        data: The feature's fields. Must be JSON-serializable.

    Returns:
        A shallow copy of `message`, with the same `id`.

    Example:
        ```python
        from langchain_core.messages import HumanMessage, add_notification_metadata

        message = add_notification_metadata(
            HumanMessage("Background task t_42 finished."),
            source="async_subagent",
            data={"task_id": "t_42", "status": "success"},
        )
        ```
    """
    kwargs = {k: v for k, v in message.additional_kwargs.items() if k != "lc_data"}
    kwargs["lc_source"] = source
    if data is not None:
        kwargs["lc_data"] = dict(data)
    return message.model_copy(update={"additional_kwargs": kwargs})


def get_notification_metadata(message: BaseMessage) -> NotificationMetadata | None:
    """Return the notification metadata of `message`.

    !!! version-added "Added in `langchain-core` 1.7.2"

    Args:
        message: The message to read.

    Returns:
        The metadata `add_notification_metadata` set, or `None` for a message
            without a string `lc_source`. `data` is a copy of `lc_data`, left out
            unless `lc_data` is a dict.
    """
    source = message.additional_kwargs.get("lc_source")
    if not isinstance(source, str):
        return None
    metadata = NotificationMetadata(source=source)
    data = message.additional_kwargs.get("lc_data")
    if isinstance(data, dict):
        metadata["data"] = dict(data)
    return metadata
