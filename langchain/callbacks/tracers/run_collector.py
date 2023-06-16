"""A tracer that collects all nested runs in a list."""
from typing import Any, List

from langchain.callbacks.tracers.base import BaseTracer
from langchain.callbacks.tracers.schemas import Run


class RunCollectorCallbackHandler(BaseTracer):
    """A tracer that collects all nested runs in a list.

    Useful for inspection and for evaluation."""

    name = "run-collector_callback_handler"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.traced_runs: List[Run] = []

    def _persist_run(self, run: Run) -> None:
        self.traced_runs.append(run)
