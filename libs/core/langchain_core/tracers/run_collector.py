"""A tracer that collects all nested runs in a list."""

from typing import Any, List, Optional, Union
from uuid import UUID

from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run


class RunCollectorCallbackHandler(BaseTracer):
    """
    A tracer that collects all nested runs in a list.

    This tracer is useful for inspection and evaluation purposes.

    Parameters
    ----------
    example_id : Optional[Union[UUID, str]], default=None
        The ID of the example being traced. It can be either a UUID or a string.
    """

    name: str = "run-collector_callback_handler"

    def __init__(
        self, example_id: Optional[Union[UUID, str]] = None, **kwargs: Any
    ) -> None:
        """
        Initialize the RunCollectorCallbackHandler.

        Parameters
        ----------
        example_id : Optional[Union[UUID, str]], default=None
            The ID of the example being traced. It can be either a UUID or a string.
        """
        super().__init__(**kwargs)
        self.example_id = (
            UUID(example_id) if isinstance(example_id, str) else example_id
        )
        self.traced_runs: List[Run] = []

    def _persist_run(self, run: Run) -> None:
        """
        Persist a run by adding it to the traced_runs list.

        Parameters
        ----------
        run : Run
            The run to be persisted.
        """
        run_ = run.copy()
        run_.reference_example_id = self.example_id
        self.traced_runs.append(run_)
