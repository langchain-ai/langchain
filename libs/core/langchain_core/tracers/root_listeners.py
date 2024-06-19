from typing import Awaitable, Callable, Optional, Union
from uuid import UUID

from langchain_core.runnables.config import (
    RunnableConfig,
    acall_func_with_variable_args,
    call_func_with_variable_args,
)
from langchain_core.tracers.base import AsyncBaseTracer, BaseTracer
from langchain_core.tracers.schemas import Run

Listener = Union[Callable[[Run], None], Callable[[Run, RunnableConfig], None]]
AsyncListener = Union[
    Callable[[Run], Awaitable[None]], Callable[[Run, RunnableConfig], Awaitable[None]]
]


class RootListenersTracer(BaseTracer):
    """Tracer that calls listeners on run start, end, and error."""

    def __init__(
        self,
        *,
        config: RunnableConfig,
        on_start: Optional[Listener],
        on_end: Optional[Listener],
        on_error: Optional[Listener],
    ) -> None:
        super().__init__(_schema_format="original+chat")

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


class AsyncRootListenersTracer(AsyncBaseTracer):
    """Async Tracer that calls listeners on run start, end, and error."""

    def __init__(
        self,
        *,
        config: RunnableConfig,
        on_start: Optional[AsyncListener],
        on_end: Optional[AsyncListener],
        on_error: Optional[AsyncListener],
    ) -> None:
        super().__init__()

        self.config = config
        self._arg_on_start = on_start
        self._arg_on_end = on_end
        self._arg_on_error = on_error
        self.root_id: Optional[UUID] = None

    async def _persist_run(self, run: Run) -> None:
        # This is a legacy method only called once for an entire run tree
        # therefore not useful here
        pass

    async def _on_run_create(self, run: Run) -> None:
        if self.root_id is not None:
            return

        self.root_id = run.id

        if self._arg_on_start is not None:
            await acall_func_with_variable_args(self._arg_on_start, run, self.config)

    async def _on_run_update(self, run: Run) -> None:
        if run.id != self.root_id:
            return

        if run.error is None:
            if self._arg_on_end is not None:
                await acall_func_with_variable_args(self._arg_on_end, run, self.config)
        else:
            if self._arg_on_error is not None:
                await acall_func_with_variable_args(
                    self._arg_on_error, run, self.config
                )
