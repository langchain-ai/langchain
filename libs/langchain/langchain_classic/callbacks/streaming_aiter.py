from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Literal, cast

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from typing_extensions import override

# TODO: If used by two LLM runs in parallel this won't work as expected


class AsyncIteratorCallbackHandler(AsyncCallbackHandler):
    """Callback handler that returns an async iterator."""

    queue: asyncio.Queue[str]

    done: asyncio.Event

    @property
    def always_verbose(self) -> bool:
        """Always verbose."""
        return True

    def __init__(self) -> None:
        """Instantiate AsyncIteratorCallbackHandler."""
        self.queue = asyncio.Queue()
        self.done = asyncio.Event()

    @override
    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        # If two calls are made in a row, this resets the state
        self.done.clear()

    @override
    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if token is not None and token != "":
            self.queue.put_nowait(token)

    @override
    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.done.set()

    @override
    async def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self.done.set()

    # TODO: implement the other methods

    async def aiter(self) -> AsyncIterator[str]:
        """Asynchronous iterator that yields tokens."""
        while not self.queue.empty() or not self.done.is_set():
            # Wait for the next token in the queue,
            # but stop waiting if the done event is set
            done, other = await asyncio.wait(
                [
                    # NOTE: If you add other tasks here, update the code below,
                    # which assumes each set has exactly one task each
                    asyncio.ensure_future(self.queue.get()),
                    asyncio.ensure_future(self.done.wait()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel the other task
            if other:
                other.pop().cancel()

            # Extract the value of the first completed task
            token_or_done = cast("str | Literal[True]", done.pop().result())

            # If the extracted value is the boolean True, the done event was set
            if token_or_done is True:
                break

            # Otherwise, the extracted value is a token, which we yield
            yield token_or_done
