"""A tracer that collects all nested runs in a list."""

from typing import Any, Optional, Union
from uuid import UUID

from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run


class RunCollectorCallbackHandler(BaseTracer):
    """Tracer that collects all nested runs in a list.

    This tracer is useful for inspection and evaluation purposes.
    """

    name: str = "run-collector_callback_handler"

    def __init__(
        self, example_id: Optional[Union[UUID, str]] = None, **kwargs: Any
    ) -> None:
        """Initialize the RunCollectorCallbackHandler.

        Args:
            example_id: The ID of the example being traced. (default: None).
                It can be either a UUID or a string.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.example_id = (
            UUID(example_id) if isinstance(example_id, str) else example_id
        )
        self.traced_runs: list[Run] = []

    def _persist_run(self, run: Run) -> None:
        """Persist a run by adding it to the traced_runs list.

        Args:
            run: The run to be persisted.
        """
        run_ = run.copy()
        run_.reference_example_id = self.example_id
        self.traced_runs.append(run_)
