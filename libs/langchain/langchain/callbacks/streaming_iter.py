from __future__ import annotations

import threading
from queue import Queue
from typing import Any, Dict, Iterator, List, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.output import LLMResult


class IteratorCallbackHandler(BaseCallbackHandler):
    """Callback handler that returns an async iterator."""

    queue: Queue[str]

    done: threading.Event

    @property
    def always_verbose(self) -> bool:
        return True

    def __init__(self) -> None:
        self.queue = Queue()
        self.done = threading.Event()

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        # If two calls are made in a row, this resets the state
        self.done.clear()

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if token is not None and token != "":
            self.queue.put_nowait(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.done.set()

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        self.done.set()

    def iter(self) -> Iterator[str]:
        while not self.done.is_set():
            try:
                yield self.queue.get(timeout=0.01)
            except Exception:
                pass
