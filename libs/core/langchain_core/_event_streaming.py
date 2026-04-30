"""Hybrid result object for `astream_events`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Coroutine, Generator

T_await = TypeVar("T_await")
"""Type returned when the result is `await`ed (e.g. `AsyncChatModelStream`)."""

T_iter = TypeVar("T_iter")
"""Type yielded when the result is async-iterated (e.g. `StreamEvent`)."""


class _AsyncEventsResult(Generic[T_await, T_iter]):
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

    Type parameters:
        T_await: The type returned by `await` (v3). Use `Never` when the
            result is iter-only.
        T_iter: The item type yielded by `async for` (v1/v2). Use `Never`
            when the result is await-only.
    """

    __slots__ = ("_awaitable", "_iterator")

    def __init__(
        self,
        *,
        awaitable: Coroutine[Any, Any, T_await] | None = None,
        iterator: AsyncIterator[T_iter] | None = None,
    ) -> None:
        self._awaitable = awaitable
        self._iterator = iterator

    def __await__(self) -> Generator[Any, None, T_await]:
        if self._awaitable is None:
            msg = (
                "This `astream_events` result is not awaitable. "
                "Use `async for` (version='v1' or 'v2'), or pass "
                "version='v3' to get an awaitable result."
            )
            raise TypeError(msg)
        return self._awaitable.__await__()

    def __aiter__(self) -> AsyncIterator[T_iter]:
        if self._iterator is None:
            msg = (
                "This `astream_events` result is not iterable. "
                "Use `await` (version='v3'), or pass version='v1' / 'v2' "
                "to get an iterable result."
            )
            raise TypeError(msg)
        return self._iterator
