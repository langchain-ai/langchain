"""Human message."""

from typing import Any, Literal, cast, overload

from langchain_core.messages import content as types
from langchain_core.messages.base import BaseMessage, BaseMessageChunk


class HumanMessage(BaseMessage):
    """Message from the user.

    A `HumanMessage` is a message that is passed in from a user to the model.

    Example:
        ```python
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content="You are a helpful assistant! Your name is Bob."),
            HumanMessage(content="What is your name?"),
        ]

        # Instantiate a chat model and invoke it with the messages
        model = ...
        print(model.invoke(messages))
        ```
    """

    type: Literal["human"] = "human"
    """The type of the message (used for serialization)."""

    @overload
    def __init__(
        self,
        content: str | list[str | dict[Any, Any]],
        **kwargs: Any,
    ) -> None: ...

    @overload
    def __init__(
        self,
        content: str | list[str | dict[Any, Any]] | None = None,
        content_blocks: list[types.ContentBlock] | None = None,
        **kwargs: Any,
    ) -> None: ...

    def __init__(
        self,
        content: str | list[str | dict[Any, Any]] | None = None,
        content_blocks: list[types.ContentBlock] | None = None,
        **kwargs: Any,
    ) -> None:
        """Specify `content` as positional arg or `content_blocks` for typing."""
        if content_blocks is not None:
            super().__init__(
                content=cast("list[str | dict[Any, Any]]", content_blocks),
                **kwargs,
            )
        else:
            super().__init__(content=content, **kwargs)


class HumanMessageChunk(HumanMessage, BaseMessageChunk):
    """Human Message chunk."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["HumanMessageChunk"] = "HumanMessageChunk"  # type: ignore[assignment]
    """The type of the message (used for serialization)."""


NOTIFICATION_HEADER = "[SYSTEM NOTIFICATION - NOT USER INPUT]"
"""First line of a notification's content."""


def create_notification_message(
    content: str | list[types.ContentBlock],
    *,
    source: str,
    data: dict[str, Any] | None = None,
    note: str | None = None,
    header: bool = True,
    id: str | None = None,
) -> HumanMessage:
    """Create a message the agent harness adds to the conversation.

    Use it for content the user didn't type, such as a background task update or
    a conversation summary. It sets these `additional_kwargs` keys, which chat
    model integrations don't send to the model:

    - `lc_synthetic`: `True`.
    - `lc_source`: the feature that added the message, e.g. `'async_subagent'`.
    - `lc_data`: that feature's fields, when given.

    Anyone who can send input to a thread can set these keys. Don't use them to
    decide whether to trust a message.

    Don't merge notifications with other messages (`merge_message_runs`, chunk
    addition); their `additional_kwargs` merge too.

    !!! warning "Beta"

        The keys and header may change.

    !!! version-added "Added in `langchain-core` 1.8.0"

    Args:
        content: The text or content blocks the model reads. Escape untrusted
            input.
        source: Which feature added the message.
        data: Fields for code that handles this source. Must be JSON-serializable.
        note: A line shown to the model under the header.
        header: Prefix the content with `NOTIFICATION_HEADER`.
        id: The message ID.

    Returns:
        A `HumanMessage`.

    Raises:
        ValueError: If `note` is given with `header=False`.

    Example:
        ```python
        from langchain_core.messages import create_notification_message

        message = create_notification_message(
            "<task-id>t_42</task-id>",
            source="async_subagent",
            data={"task_id": "t_42", "status": "success"},
            note="A background task you started has an update.",
        )
        ```
    """
    if note and not header:
        msg = "`note` requires `header=True`."
        raise ValueError(msg)
    kwargs: dict[str, Any] = {"lc_synthetic": True, "lc_source": source}
    if data is not None:
        kwargs["lc_data"] = dict(data)
    if header:
        intro = f"{NOTIFICATION_HEADER}\n{note}" if note else NOTIFICATION_HEADER
        if isinstance(content, str):
            content = f"{intro}\n{content}"
        else:
            content = [types.TextContentBlock(type="text", text=intro), *content]
    if isinstance(content, str):
        return HumanMessage(content=content, additional_kwargs=kwargs, id=id)
    return HumanMessage(content_blocks=content, additional_kwargs=kwargs, id=id)
