"""Base interfaces for tracing runs."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.tracers.schemas import Run, RunTypeEnum
from langchain.schema import LLMResult


class TracerException(Exception):
    """Base class for exceptions in tracers module."""


class BaseTracer(BaseCallbackHandler, ABC):
    """Base interface for tracers."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run_map: Dict[str, Run] = {}

    @staticmethod
    def _add_child_run(
        parent_run: Run,
        child_run: Run,
    ) -> None:
        """Add child run to a chain run or tool run."""
        parent_run.child_runs.append(child_run)

    @abstractmethod
    def _persist_run(self, run: Run) -> None:
        """Persist a run."""

    def _start_trace(self, run: Run) -> None:
        """Start a trace for a run."""
        if run.parent_run_id:
            parent_run = self.run_map[str(run.parent_run_id)]
            if parent_run:
                self._add_child_run(parent_run, run)
            else:
                raise TracerException(
                    f"Parent run with UUID {run.parent_run_id} not found."
                )
        self.run_map[str(run.id)] = run

    def _end_trace(self, run: Run) -> None:
        """End a trace for a run."""
        if not run.parent_run_id:
            self._persist_run(run)
        else:
            parent_run = self.run_map.get(str(run.parent_run_id))
            if parent_run is None:
                raise TracerException(
                    f"Parent run with UUID {run.parent_run_id} not found."
                )
            if (
                run.child_execution_order is not None
                and parent_run.child_execution_order is not None
                and run.child_execution_order > parent_run.child_execution_order
            ):
                parent_run.child_execution_order = run.child_execution_order
        self.run_map.pop(str(run.id))

    def _get_execution_order(self, parent_run_id: Optional[str] = None) -> int:
        """Get the execution order for a run."""
        if parent_run_id is None:
            return 1

        parent_run = self.run_map.get(parent_run_id)
        if parent_run is None:
            raise TracerException(f"Parent run with UUID {parent_run_id} not found.")
        if parent_run.child_execution_order is None:
            raise TracerException(
                f"Parent run with UUID {parent_run_id} has no child execution order."
            )

        return parent_run.child_execution_order + 1

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Start a trace for an LLM run."""
        parent_run_id_ = str(parent_run_id) if parent_run_id else None
        execution_order = self._get_execution_order(parent_run_id_)
        llm_run = Run(
            id=run_id,
            parent_run_id=parent_run_id,
            serialized=serialized,
            inputs={"prompts": prompts},
            extra=kwargs,
            start_time=datetime.utcnow(),
            execution_order=execution_order,
            child_execution_order=execution_order,
            run_type=RunTypeEnum.llm,
        )
        self._start_trace(llm_run)
        self._on_llm_start(llm_run)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        """End a trace for an LLM run."""
        if not run_id:
            raise TracerException("No run_id provided for on_llm_end callback.")

        run_id_ = str(run_id)
        llm_run = self.run_map.get(run_id_)
        if llm_run is None or llm_run.run_type != RunTypeEnum.llm:
            raise TracerException("No LLM Run found to be traced")
        llm_run.outputs = response.dict()
        llm_run.end_time = datetime.utcnow()
        self._end_trace(llm_run)
        self._on_llm_end(llm_run)

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Handle an error for an LLM run."""
        if not run_id:
            raise TracerException("No run_id provided for on_llm_error callback.")

        run_id_ = str(run_id)
        llm_run = self.run_map.get(run_id_)
        if llm_run is None or llm_run.run_type != RunTypeEnum.llm:
            raise TracerException("No LLM Run found to be traced")
        llm_run.error = repr(error)
        llm_run.end_time = datetime.utcnow()
        self._end_trace(llm_run)
        self._on_chain_error(llm_run)

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Start a trace for a chain run."""
        parent_run_id_ = str(parent_run_id) if parent_run_id else None
        execution_order = self._get_execution_order(parent_run_id_)
        chain_run = Run(
            id=run_id,
            parent_run_id=parent_run_id,
            serialized=serialized,
            inputs=inputs,
            extra=kwargs,
            start_time=datetime.utcnow(),
            execution_order=execution_order,
            child_execution_order=execution_order,
            child_runs=[],
            run_type=RunTypeEnum.chain,
        )
        self._start_trace(chain_run)
        self._on_chain_start(chain_run)

    def on_chain_end(
        self, outputs: Dict[str, Any], *, run_id: UUID, **kwargs: Any
    ) -> None:
        """End a trace for a chain run."""
        if not run_id:
            raise TracerException("No run_id provided for on_chain_end callback.")
        chain_run = self.run_map.get(str(run_id))
        if chain_run is None or chain_run.run_type != RunTypeEnum.chain:
            raise TracerException("No chain Run found to be traced")

        chain_run.outputs = outputs
        chain_run.end_time = datetime.utcnow()
        self._end_trace(chain_run)
        self._on_chain_end(chain_run)

    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Handle an error for a chain run."""
        if not run_id:
            raise TracerException("No run_id provided for on_chain_error callback.")
        chain_run = self.run_map.get(str(run_id))
        if chain_run is None or chain_run.run_type != RunTypeEnum.chain:
            raise TracerException("No chain Run found to be traced")

        chain_run.error = repr(error)
        chain_run.end_time = datetime.utcnow()
        self._end_trace(chain_run)
        self._on_chain_error(chain_run)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Start a trace for a tool run."""
        parent_run_id_ = str(parent_run_id) if parent_run_id else None
        execution_order = self._get_execution_order(parent_run_id_)
        tool_run = Run(
            id=run_id,
            parent_run_id=parent_run_id,
            serialized=serialized,
            inputs={"input": input_str},
            extra=kwargs,
            start_time=datetime.utcnow(),
            execution_order=execution_order,
            child_execution_order=execution_order,
            child_runs=[],
            run_type=RunTypeEnum.tool,
        )
        self._start_trace(tool_run)
        self._on_tool_start(tool_run)

    def on_tool_end(self, output: str, *, run_id: UUID, **kwargs: Any) -> None:
        """End a trace for a tool run."""
        if not run_id:
            raise TracerException("No run_id provided for on_tool_end callback.")
        tool_run = self.run_map.get(str(run_id))
        if tool_run is None or tool_run.run_type != RunTypeEnum.tool:
            raise TracerException("No tool Run found to be traced")

        tool_run.outputs = {"output": output}
        tool_run.end_time = datetime.utcnow()
        self._end_trace(tool_run)
        self._on_tool_end(tool_run)

    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Handle an error for a tool run."""
        if not run_id:
            raise TracerException("No run_id provided for on_tool_error callback.")
        tool_run = self.run_map.get(str(run_id))
        if tool_run is None or tool_run.run_type != RunTypeEnum.tool:
            raise TracerException("No tool Run found to be traced")

        tool_run.error = repr(error)
        tool_run.end_time = datetime.utcnow()
        self._end_trace(tool_run)
        self._on_tool_error(tool_run)

    def __deepcopy__(self, memo: dict) -> BaseTracer:
        """Deepcopy the tracer."""
        return self

    def __copy__(self) -> BaseTracer:
        """Copy the tracer."""
        return self

    def _on_llm_start(self, run: Run) -> None:
        """Process the LLM Run upon start."""

    def _on_llm_end(self, run: Run) -> None:
        """Process the LLM Run."""

    def _on_llm_error(self, run: Run) -> None:
        """Process the LLM Run upon error."""

    def _on_chain_start(self, run: Run) -> None:
        """Process the Chain Run upon start."""

    def _on_chain_end(self, run: Run) -> None:
        """Process the Chain Run."""

    def _on_chain_error(self, run: Run) -> None:
        """Process the Chain Run upon error."""

    def _on_tool_start(self, run: Run) -> None:
        """Process the Tool Run upon start."""

    def _on_tool_end(self, run: Run) -> None:
        """Process the Tool Run."""

    def _on_tool_error(self, run: Run) -> None:
        """Process the Tool Run upon error."""

    def _on_chat_model_start(self, run: Run) -> None:
        """Process the Chat Model Run upon start."""
