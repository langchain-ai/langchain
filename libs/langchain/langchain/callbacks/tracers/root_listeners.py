from typing import Callable, Optional, Union
from uuid import UUID

from langchain.callbacks.tracers.base import BaseTracer
from langchain.callbacks.tracers.schemas import Run
from langchain.schema.runnable.config import (
    RunnableConfig,
    call_func_with_variable_args,
)

Listener = Union[Callable[[Run], None], Callable[[Run, RunnableConfig], None]]


class RootListenersTracer(BaseTracer):
    def __init__(
        self,
        *,
        config: RunnableConfig,
        on_start: Optional[Listener],
        on_end: Optional[Listener],
        on_error: Optional[Listener],
    ) -> None:
        super().__init__()

        self.config = config
        self._arg_on_start = on_start
        self._arg_on_end = on_end
        self._arg_on_error = on_error
        self.root_id: Optional[UUID] = None

    def _persist_run(self, run: Run) -> None:
        # This is a legacy method only called once for an entire run tree
        # therefore not useful here
        pass

    def _on_run_create(self, run: Run) -> None:
        if self.root_id is not None:
            return

        self.root_id = run.id

        if self._arg_on_start is not None:
            call_func_with_variable_args(self._arg_on_start, run, self.config)

    def _on_run_update(self, run: Run) -> None:
        if run.id != self.root_id:
            return

        if run.error is None:
            if self._arg_on_end is not None:
                call_func_with_variable_args(self._arg_on_end, run, self.config)
        else:
            if self._arg_on_error is not None:
                call_func_with_variable_args(self._arg_on_error, run, self.config)
