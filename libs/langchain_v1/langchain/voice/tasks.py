"""Parallel, interruptible background task orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from langchain.voice.events import (
    TaskCancelledVoiceEvent,
    TaskCompletedVoiceEvent,
    TaskCreatedVoiceEvent,
    TaskFailedVoiceEvent,
    TaskStartedVoiceEvent,
    TaskUpdatedVoiceEvent,
    VoiceEvent,
)
from langchain.voice.tracing import trace_task

if TYPE_CHECKING:
    from langchain.voice.brain import GraphBrain

MAX_INSTRUCTION_CHARS = 16_000
DEFAULT_MAX_ACTIVE_TASKS = 16
MAX_TASKS_PER_SESSION = 1_024
EventSink = Callable[[VoiceEvent], Awaitable[None]]


class TaskStatus(str, Enum):
    """Lifecycle state for one background task revision."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskError(ValueError):
    """A safe task error that may be returned to the connected client."""

    code = "task_error"


class TaskNotFoundError(TaskError):
    """Raised when a task does not belong to the current session."""

    code = "task_not_found"


class TaskStateError(TaskError):
    """Raised when an operation is invalid for a task's current state."""

    code = "invalid_task_state"


class TaskLimitError(TaskError):
    """Raised when a session exceeds its configured task limits."""

    code = "task_limit_reached"


@dataclass(slots=True)
class TaskRecord:
    """Mutable runtime state for one logical background task."""

    task_id: str
    thread_id: str
    instruction: str
    revision: int = 1
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None
    runner: asyncio.Task[None] | None = field(default=None, repr=False)
    operation_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


def _validated_instruction(instruction: object) -> str:
    if not isinstance(instruction, str):
        msg = "instruction must be a string"
        raise TaskError(msg)
    normalized = instruction.strip()
    if not normalized:
        msg = "instruction cannot be empty"
        raise TaskError(msg)
    if len(normalized) > MAX_INSTRUCTION_CHARS:
        msg = f"instruction exceeds the {MAX_INSTRUCTION_CHARS}-character limit"
        raise TaskError(msg)
    return normalized


class TaskManager:
    """Own all background work for exactly one connected voice session."""

    def __init__(
        self,
        brain: GraphBrain,
        event_sink: EventSink,
        *,
        max_active_tasks: int = DEFAULT_MAX_ACTIVE_TASKS,
    ) -> None:
        """Initialize a session-scoped task manager."""
        if max_active_tasks < 1:
            msg = "max_active_tasks must be positive"
            raise ValueError(msg)
        self._brain = brain
        self._event_sink = event_sink
        self._max_active_tasks = max_active_tasks
        self._records: dict[str, TaskRecord] = {}
        self._state_lock = asyncio.Lock()
        self._closed = False

    async def create_task(self, instruction: str) -> str:
        """Start a new independent task and return its opaque identifier."""
        instruction = _validated_instruction(instruction)
        async with self._state_lock:
            self._ensure_open()
            active = sum(
                record.status in {TaskStatus.PENDING, TaskStatus.RUNNING}
                for record in self._records.values()
            )
            if active >= self._max_active_tasks:
                msg = "too many background tasks are active"
                raise TaskLimitError(msg)
            if len(self._records) >= MAX_TASKS_PER_SESSION:
                msg = "too many tasks were created in this session"
                raise TaskLimitError(msg)
            task_id = str(uuid4())
            record = TaskRecord(
                task_id=task_id,
                thread_id=str(uuid4()),
                instruction=instruction,
            )
            self._records[task_id] = record

        await self._event_sink(TaskCreatedVoiceEvent(task_id=task_id, revision=1))
        record.runner = asyncio.create_task(
            self._execute(record, revision=1), name=f"langchain-voice-task-{task_id}"
        )
        return task_id

    async def update_task(self, task_id: str, instruction: str) -> None:
        """Cancel the active revision and run replacement work on the same thread."""
        instruction = _validated_instruction(instruction)
        record = await self._get_record(task_id)
        async with record.operation_lock:
            async with self._state_lock:
                self._ensure_open()
                if record.status is TaskStatus.CANCELLED:
                    msg = "a cancelled task cannot be updated"
                    raise TaskStateError(msg)
                record.revision += 1
                revision = record.revision
                previous_runner = record.runner
                record.runner = None
                record.instruction = instruction
                record.result = None
                record.status = TaskStatus.PENDING

            if previous_runner is not None and not previous_runner.done():
                previous_runner.cancel()
                if previous_runner is not asyncio.current_task():
                    await asyncio.gather(previous_runner, return_exceptions=True)

            await self._event_sink(TaskUpdatedVoiceEvent(task_id=task_id, revision=revision))
            record.runner = asyncio.create_task(
                self._execute(record, revision=revision),
                name=f"langchain-voice-task-{task_id}-r{revision}",
            )

    async def cancel_task(self, task_id: str) -> None:
        """Cancel a task and prevent future updates to it."""
        record = await self._get_record(task_id)
        async with record.operation_lock:
            async with self._state_lock:
                self._ensure_open()
                if record.status is TaskStatus.CANCELLED:
                    return
                record.revision += 1
                runner = record.runner
                record.runner = None
                record.status = TaskStatus.CANCELLED
                record.result = None
                revision = record.revision

            if runner is not None and not runner.done():
                runner.cancel()
                if runner is not asyncio.current_task():
                    await asyncio.gather(runner, return_exceptions=True)
            await self._event_sink(TaskCancelledVoiceEvent(task_id=task_id, revision=revision))

    async def get_status(self, task_id: str) -> TaskStatus:
        """Return the current lifecycle state for a task."""
        return (await self._get_record(task_id)).status

    async def aclose(self) -> None:
        """Cancel every unfinished task owned by this manager."""
        async with self._state_lock:
            if self._closed:
                return
            self._closed = True
            runners: list[asyncio.Task[None]] = []
            for record in self._records.values():
                record.revision += 1
                record.status = TaskStatus.CANCELLED
                if record.runner is not None and not record.runner.done():
                    record.runner.cancel()
                    runners.append(record.runner)
                record.runner = None
        if runners:
            await asyncio.gather(*runners, return_exceptions=True)

    async def _get_record(self, task_id: str) -> TaskRecord:
        if not isinstance(task_id, str) or not task_id:
            msg = "task does not exist in this session"
            raise TaskNotFoundError(msg)
        async with self._state_lock:
            record = self._records.get(task_id)
        if record is None:
            msg = "task does not exist in this session"
            raise TaskNotFoundError(msg)
        return record

    async def _execute(self, record: TaskRecord, *, revision: int) -> None:
        async with self._state_lock:
            if self._closed or record.revision != revision:
                return
            record.status = TaskStatus.RUNNING
            instruction = record.instruction
        await self._event_sink(TaskStartedVoiceEvent(task_id=record.task_id, revision=revision))

        try:
            result = await self._run_brain(record, revision, instruction)
        except asyncio.CancelledError:
            raise
        # User graphs may raise arbitrary provider or tool exceptions. Keep the
        # session alive and expose only a stable, non-sensitive failure.
        except Exception:
            async with self._state_lock:
                if self._closed or record.revision != revision:
                    return
                record.status = TaskStatus.FAILED
                record.runner = None
            await self._event_sink(
                TaskFailedVoiceEvent(
                    task_id=record.task_id,
                    revision=revision,
                    error="The background agent could not complete the task.",
                )
            )
            return

        async with self._state_lock:
            if self._closed or record.revision != revision:
                return
            record.status = TaskStatus.COMPLETED
            record.result = result
            record.runner = None
        await self._event_sink(
            TaskCompletedVoiceEvent(
                task_id=record.task_id,
                revision=revision,
                result=result,
            )
        )

    async def _run_brain(self, record: TaskRecord, revision: int, instruction: str) -> str:
        """Run one graph revision and close its trace with a curated outcome."""
        result: str | None = None
        pending_error: BaseException | None = None
        async with trace_task(
            task_id=record.task_id,
            thread_id=record.thread_id,
            revision=revision,
            instruction=instruction,
        ) as task_trace:
            try:
                result = await self._brain.run(instruction, thread_id=record.thread_id)
            except asyncio.CancelledError as exc:
                task_trace.cancelled()
                pending_error = exc
            except Exception as exc:
                task_trace.failed()
                pending_error = exc
            else:
                task_trace.completed(result)

        if pending_error is not None:
            raise pending_error
        return cast("str", result)

    def _ensure_open(self) -> None:
        if self._closed:
            msg = "the voice session is closed"
            raise TaskStateError(msg)


class TaskTools:
    """The scoped tools exposed to one session's conversation layer."""

    def __init__(self, manager: TaskManager) -> None:
        """Bind task operations to one session-owned manager."""
        self._manager = manager

    async def create_task(self, instruction: str) -> str:
        """Create an independent background task."""
        return await self._manager.create_task(instruction)

    async def update_task(self, task_id: str, instruction: str) -> None:
        """Replace the work for a background task."""
        await self._manager.update_task(task_id, instruction)

    async def cancel_task(self, task_id: str) -> None:
        """Cancel a background task."""
        await self._manager.cancel_task(task_id)
