from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List, Literal, Union, cast, Optional

from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema.output import LLMResult


class Run:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.done = asyncio.Event()


class AsyncIteratorCallbackHandler(AsyncCallbackHandler):
    """Callback handler that returns an async iterator."""

    runs_queue: asyncio.Queue[Run]
    current_run: Optional[Run] = None
    all_done: asyncio.Event
    pending_runs: int

    @property
    def always_verbose(self) -> bool:
        return True

    def __init__(self) -> None:
        self.runs_queue = asyncio.Queue()
        self.all_done = asyncio.Event()
        self.pending_runs = 0

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        # If two calls are made in a row, this resets the state
        self.current_run = Run()
        self.runs_queue.put_nowait(self.current_run)
        self.pending_runs += 1

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if token is not None and token != "":
            self.current_run.queue.put_nowait(token)

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.current_run.done.set()
        # When a Run object ends, decrement pending_runs
        self.pending_runs -= 1
        # If there are no running Run objects
        # wait for a while for new Run objects to have a chance to start
        # and if there are still no running Run objects
        # set the all_done event
        if self.pending_runs == 0:
            await asyncio.sleep(1)
            if self.pending_runs == 0:
                self.all_done.set()

    async def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        self.done.set()

    # TODO implement the other methods

    async def aiter(self) -> AsyncIterator[str]:
        while not self.all_done.is_set():
            if self.current_run is None or self.current_run.done.is_set():
                try:
                    self.current_run = await self.runs_queue.get()
                except asyncio.QueueEmpty:
                    break
            while (
                not self.current_run.queue.empty() or not self.current_run.done.is_set()
            ):
                # Wait for the next token in the queue,
                # but stop waiting if the done event is set
                done, other = await asyncio.wait(
                    [
                        # NOTE: If you add other tasks here, update the code below,
                        # which assumes each set has exactly one task each
                        asyncio.ensure_future(self.current_run.queue.get()),
                        asyncio.ensure_future(self.current_run.done.wait()),
                    ],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Cancel the other task
                if other:
                    other.pop().cancel()

                # Extract the value of the first completed task
                token_or_done = cast(Union[str, Literal[True]], done.pop().result())

                # If the extracted value is the boolean True, the done event was set
                if token_or_done is True:
                    break

                # Otherwise, the extracted value is a token, which we yield
                yield token_or_done
