"""Optional LangSmith tracing for background task lifecycles."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

MAX_TRACE_TEXT_CHARS = 12_000
logger = logging.getLogger(__name__)


class _TraceRun(Protocol):
    def end(self, *, outputs: dict[str, Any]) -> None: ...


def _bounded(text: str) -> str:
    if len(text) <= MAX_TRACE_TEXT_CHARS:
        return text
    return text[:MAX_TRACE_TEXT_CHARS] + " …[truncated]"


class TaskTrace:
    """Finish one optional task span with a stable lifecycle outcome."""

    def __init__(self, run: _TraceRun | None, *, task_id: str, revision: int) -> None:
        """Initialize a lifecycle wrapper around an optional LangSmith run."""
        self._run = run
        self._task_id = task_id
        self._revision = revision
        self._finished = False

    def completed(self, result: str) -> None:
        """Finish the trace with a bounded successful result."""
        self._finish(status="completed", result=_bounded(result))

    def failed(self) -> None:
        """Finish the trace with a stable, non-sensitive failure."""
        self._finish(
            status="failed",
            error="The background agent could not complete the task.",
        )

    def cancelled(self) -> None:
        """Finish the trace as cancelled."""
        self._finish(status="cancelled")

    def _finish(self, *, status: str, **output: Any) -> None:
        if self._run is None or self._finished:
            return
        self._finished = True
        try:
            self._run.end(
                outputs={
                    "task_id": self._task_id,
                    "revision": self._revision,
                    "status": status,
                    **output,
                }
            )
        except Exception:
            logger.warning("LangChain Voice task tracing failed while recording an outcome")


def _load_trace_context() -> Any | None:
    try:
        from langsmith import trace  # noqa: PLC0415 - LangSmith is optional
    except ImportError:
        return None
    return trace


@asynccontextmanager
async def trace_task(
    *,
    task_id: str,
    thread_id: str,
    revision: int,
    instruction: str,
) -> AsyncIterator[TaskTrace]:
    """Trace a task revision under the active voice-session trace, if present."""
    trace_context = _load_trace_context()
    if trace_context is None:
        yield TaskTrace(None, task_id=task_id, revision=revision)
        return

    context = trace_context(
        "langchain_voice_task",
        run_type="chain",
        inputs={"instruction": _bounded(instruction)},
        tags=["langchain-voice", "background-task"],
        metadata={
            "task_id": task_id,
            "thread_id": thread_id,
            "revision": revision,
        },
    )
    try:
        run = await context.__aenter__()
    except Exception:
        logger.warning("LangChain Voice task tracing failed while starting a span")
        yield TaskTrace(None, task_id=task_id, revision=revision)
        return

    try:
        yield TaskTrace(run, task_id=task_id, revision=revision)
    finally:
        try:
            await context.__aexit__(None, None, None)
        except Exception:
            logger.warning("LangChain Voice task tracing failed while closing a span")
