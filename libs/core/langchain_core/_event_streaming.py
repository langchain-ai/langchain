"""Hybrid result object for `astream_events`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Coroutine, Generator


class _AsyncEventsResult:
    """Hybrid object returned by `BaseChatModel.astream_events`.

    Supports two distinct calling patterns based on `version`:

    - `version="v1"` / `"v2"`: async-iterate to get `StreamEvent` items.
      ```python
      async for ev in chat_model.astream_events(input, version="v2"):
          ...
      ```
    - `version="v3"`: `await` to get the typed stream object
      (`AsyncChatModelStream`).
      ```python
      stream = await chat_model.astream_events(input, version="v3")
      async for chunk in stream.text:
          ...
      ```

    Mixing patterns (e.g., awaiting a v2 result, or async-for'ing a v3
    result without first iterating the stream) raises `TypeError`.
    """

    __slots__ = ("_awaitable", "_iterator")

    def __init__(
        self,
        *,
        awaitable: Coroutine[Any, Any, Any] | None = None,
        iterator: AsyncIterator[Any] | None = None,
    ) -> None:
        self._awaitable = awaitable
        self._iterator = iterator

    def __await__(self) -> Generator[Any, None, Any]:
        if self._awaitable is None:
            msg = (
                "This `astream_events` result is not awaitable. "
                "Use `async for` (version='v1' or 'v2'), or pass "
                "version='v3' to get an awaitable result."
            )
            raise TypeError(msg)
        return self._awaitable.__await__()

    def __aiter__(self) -> AsyncIterator[Any]:
        if self._iterator is None:
            msg = (
                "This `astream_events` result is not iterable. "
                "Use `await` (version='v3'), or pass version='v1' / 'v2' "
                "to get an iterable result."
            )
            raise TypeError(msg)
        return self._iterator
