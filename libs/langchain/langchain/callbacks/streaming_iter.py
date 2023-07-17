from __future__ import annotations

from queue import Queue
from typing import Any, Iterator, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult


class IteratorCallbackHandler(BaseCallbackHandler):
    """Callback handler that returns an async iterator."""

    queue: Queue[str]

    @property
    def always_verbose(self) -> bool:
        return True

    def __init__(self) -> None:
        self.queue = Queue()

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if token is not None and token != "":
            self.queue.put_nowait(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        while not self.queue.empty():
            self.queue.get()

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        while not self.queue.empty():
            self.queue.get()

    def iter(self) -> Iterator[str]:
        while not self.queue.empty():
            yield self.queue.get()
