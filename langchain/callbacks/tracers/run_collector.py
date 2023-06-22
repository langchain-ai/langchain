"""A tracer that collects all nested runs in a list."""
from typing import Any, List, Optional, Union
from uuid import UUID

from langchain.callbacks.tracers.base import BaseTracer
from langchain.callbacks.tracers.schemas import Run


class RunCollectorCallbackHandler(BaseTracer):
    """A tracer that collects all nested runs in a list.

    Useful for inspection and for evaluation."""

    name = "run-collector_callback_handler"

    def __init__(
        self, example_id: Optional[Union[UUID, str]] = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.example_id = (
            UUID(example_id) if isinstance(example_id, str) else example_id
        )
        self.traced_runs: List[Run] = []

    def _persist_run(self, run: Run) -> None:
        run_ = run.copy()
        run_.reference_example_id = self.example_id
        self.traced_runs.append(run_)
