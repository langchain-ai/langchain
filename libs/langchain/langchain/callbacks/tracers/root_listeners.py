from typing import Callable, Optional

from langchain.callbacks.tracers.base import BaseTracer
from langchain.callbacks.tracers.schemas import Run


class RootListenersTracer(BaseTracer):
    def __init__(
        self,
        *,
        on_start: Optional[Callable[[Run], None]],
        on_end: Optional[Callable[[Run], None]],
        on_error: Optional[Callable[[Run], None]]
    ) -> None:
        super().__init__()

        self._arg_on_start = on_start
        self._arg_on_end = on_end
        self._arg_on_error = on_error

    def _persist_run(self, run: Run) -> None:
        # This is a legacy method only called once for an entire run tree
        # therefore not useful here
        pass

    def _on_run_create(self, run: Run) -> None:
        if run.parent_run_id is not None:
            return

        if self._arg_on_start is not None:
            self._arg_on_start(run)

    def _on_run_update(self, run: Run) -> None:
        if run.parent_run_id is not None:
            return

        if run.error is None:
            if self._arg_on_end is not None:
                self._arg_on_end(run)
        else:
            if self._arg_on_error is not None:
                self._arg_on_error(run)
