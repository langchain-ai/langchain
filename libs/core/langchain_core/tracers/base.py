"""Base interfaces for tracing runs."""
from __future__ import annotations

import logging
import sys
import traceback
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Union,
    cast,
)
from uuid import UUID

from tenacity import RetryCallState

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.exceptions import TracerException
from langchain_core.load import dumpd
from langchain_core.messages import BaseMessage
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    GenerationChunk,
    LLMResult,
)
from langchain_core.tracers.schemas import Run

if TYPE_CHECKING:
    from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class BaseTracer(BaseCallbackHandler, ABC):
    """Base interface for tracers."""

    def __init__(
        self,
        *,
        _schema_format: Literal["original", "streaming_events"] = "original",
        **kwargs: Any,
    ) -> None:
        """Initialize the tracer.

        Args:
            _schema_format: Primarily changes how the inputs and outputs are
                handled. For internal use only. This API will change.
                - 'original' is the format used by all current tracers.
                   This format is slightly inconsistent with respect to inputs
                   and outputs.
                - 'streaming_events' is used for supporting streaming events,
                   for internal usage. It will likely change in the future, or
                   be deprecated entirely in favor of a dedicated async tracer
                   for streaming events.
            kwargs: Additional keyword arguments that will be passed to
                    the super class.
        """
        super().__init__(**kwargs)
        self._schema_format = _schema_format  # For internal use only API will change.
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

    @staticmethod
    def _get_stacktrace(error: BaseException) -> str:
        """Get the stacktrace of the parent error."""
        msg = repr(error)
        try:
            if sys.version_info < (3, 10):
                tb = traceback.format_exception(
                    error.__class__, error, error.__traceback__
                )
            else:
                tb = traceback.format_exception(error)
            return (msg + "\n\n".join(tb)).strip()
        except:  # noqa: E722
            return msg

    def _start_trace(self, run: Run) -> None:
        """Start a trace for a run."""
        current_dotted_order = run.start_time.strftime("%Y%m%dT%H%M%S%fZ") + str(run.id)
        if run.parent_run_id:
            parent_run = self.run_map.get(str(run.parent_run_id))
            if parent_run:
                self._add_child_run(parent_run, run)
                parent_run.child_execution_order = max(
                    parent_run.child_execution_order, run.child_execution_order
                )
                run.trace_id = parent_run.trace_id
                if parent_run.dotted_order:
                    run.dotted_order = (
                        parent_run.dotted_order + "." + current_dotted_order
                    )
                else:
                    # Something wrong with tracer parent run has no dotted_order
                    logger.debug(
                        f"Parent run with UUID {run.parent_run_id} has no dotted_order."
                    )
            else:
                # Something wrong with tracer, parent run not found
                # Calculate the trace_id and dotted_order server side
                logger.debug(f"Parent run with UUID {run.parent_run_id} not found.")
        else:
            run.trace_id = run.id
            run.dotted_order = current_dotted_order
        self.run_map[str(run.id)] = run
        self._on_run_create(run)

    def _end_trace(self, run: Run) -> None:
        """End a trace for a run."""
        if not run.parent_run_id:
            self._persist_run(run)
        else:
            parent_run = self.run_map.get(str(run.parent_run_id))
            if parent_run is None:
                logger.debug(f"Parent run with UUID {run.parent_run_id} not found.")
            elif (
                run.child_execution_order is not None
                and parent_run.child_execution_order is not None
                and run.child_execution_order > parent_run.child_execution_order
            ):
                parent_run.child_execution_order = run.child_execution_order
        self.run_map.pop(str(run.id))
        self._on_run_update(run)

    def _get_execution_order(self, parent_run_id: Optional[str] = None) -> int:
        """Get the execution order for a run."""
        if parent_run_id is None:
            return 1

        parent_run = self.run_map.get(parent_run_id)
        if parent_run is None:
            logger.debug(f"Parent run with UUID {parent_run_id} not found.")
            return 1
        if parent_run.child_execution_order is None:
            raise TracerException(
                f"Parent run with UUID {parent_run_id} has no child execution order."
            )

        return parent_run.child_execution_order + 1

    def _get_run(
        self, run_id: UUID, run_type: Union[str, Set[str], None] = None
    ) -> Run:
        try:
            run = self.run_map[str(run_id)]
        except KeyError as exc:
            raise TracerException(f"No indexed run ID {run_id}.") from exc

        if isinstance(run_type, str):
            run_types: Union[Set[str], None] = {run_type}
        else:
            run_types = run_type
        if run_types is not None and run.run_type not in run_types:
            raise TracerException(
                f"Found {run.run_type} run at ID {run_id}, "
                f"but expected {run_types} run."
            )
        return run

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        tags: Optional[List[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Run:
        """Start a trace for an LLM run."""
        if self._schema_format != "streaming_events":
            # Please keep this un-implemented for backwards compatibility.
            # When it's unimplemented old tracers that use the "original" format
            # fallback on the on_llm_start method implementation if they
            # find that the on_chat_model_start method is not implemented.
            # This can eventually be cleaned up by writing a "modern" tracer
            # that has all the updated schema changes corresponding to
            # the "streaming_events" format.
            raise NotImplementedError(
                f"Chat model tracing is not supported in "
                f"for {self._schema_format} format."
            )
        parent_run_id_ = str(parent_run_id) if parent_run_id else None
        execution_order = self._get_execution_order(parent_run_id_)
        start_time = datetime.now(timezone.utc)
        if metadata:
            kwargs.update({"metadata": metadata})
        chat_model_run = Run(
            id=run_id,
            parent_run_id=parent_run_id,
            serialized=serialized,
            inputs={"messages": [[dumpd(msg) for msg in batch] for batch in messages]},
            extra=kwargs,
            events=[{"name": "start", "time": start_time}],
            start_time=start_time,
            execution_order=execution_order,
            child_execution_order=execution_order,
            # WARNING: This is valid ONLY for streaming_events.
            # run_type="llm" is what's used by virtually all tracers.
            # Changing this to "chat_model" may break triggering on_llm_start
            run_type="chat_model",
            tags=tags,
            name=name,  # type: ignore[arg-type]
        )
        self._start_trace(chat_model_run)
        self._on_chat_model_start(chat_model_run)
        return chat_model_run

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        tags: Optional[List[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Run:
        """Start a trace for an LLM run."""
        parent_run_id_ = str(parent_run_id) if parent_run_id else None
        execution_order = self._get_execution_order(parent_run_id_)
        start_time = datetime.now(timezone.utc)
        if metadata:
            kwargs.update({"metadata": metadata})
        llm_run = Run(
            id=run_id,
            parent_run_id=parent_run_id,
            serialized=serialized,
            # TODO: Figure out how to expose kwargs here
            inputs={"prompts": prompts},
            extra=kwargs,
            events=[{"name": "start", "time": start_time}],
            start_time=start_time,
            execution_order=execution_order,
            child_execution_order=execution_order,
            run_type="llm",
            tags=tags or [],
            name=name,  # type: ignore[arg-type]
        )
        self._start_trace(llm_run)
        self._on_llm_start(llm_run)
        return llm_run

    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Run:
        """Run on new LLM token. Only available when streaming is enabled."""
        # "chat_model" is only used for the experimental new streaming_events format.
        # This change should not affect any existing tracers.
        llm_run = self._get_run(run_id, run_type={"llm", "chat_model"})
        event_kwargs: Dict[str, Any] = {"token": token}
        if chunk:
            event_kwargs["chunk"] = chunk
        llm_run.events.append(
            {
                "name": "new_token",
                "time": datetime.now(timezone.utc),
                "kwargs": event_kwargs,
            },
        )
        self._on_llm_new_token(llm_run, token, chunk)
        return llm_run

    def on_retry(
        self,
        retry_state: RetryCallState,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> Run:
        llm_run = self._get_run(run_id)
        retry_d: Dict[str, Any] = {
            "slept": retry_state.idle_for,
            "attempt": retry_state.attempt_number,
        }
        if retry_state.outcome is None:
            retry_d["outcome"] = "N/A"
        elif retry_state.outcome.failed:
            retry_d["outcome"] = "failed"
            exception = retry_state.outcome.exception()
            retry_d["exception"] = str(exception)
            retry_d["exception_type"] = exception.__class__.__name__
        else:
            retry_d["outcome"] = "success"
            retry_d["result"] = str(retry_state.outcome.result())
        llm_run.events.append(
            {
                "name": "retry",
                "time": datetime.now(timezone.utc),
                "kwargs": retry_d,
            },
        )
        return llm_run

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> Run:
        """End a trace for an LLM run."""
        # "chat_model" is only used for the experimental new streaming_events format.
        # This change should not affect any existing tracers.
        llm_run = self._get_run(run_id, run_type={"llm", "chat_model"})
        llm_run.outputs = response.dict()
        for i, generations in enumerate(response.generations):
            for j, generation in enumerate(generations):
                output_generation = llm_run.outputs["generations"][i][j]
                if "message" in output_generation:
                    output_generation["message"] = dumpd(
                        cast(ChatGeneration, generation).message
                    )
        llm_run.end_time = datetime.now(timezone.utc)
        llm_run.events.append({"name": "end", "time": llm_run.end_time})
        self._end_trace(llm_run)
        self._on_llm_end(llm_run)
        return llm_run

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> Run:
        """Handle an error for an LLM run."""
        # "chat_model" is only used for the experimental new streaming_events format.
        # This change should not affect any existing tracers.
        llm_run = self._get_run(run_id, run_type={"llm", "chat_model"})
        llm_run.error = self._get_stacktrace(error)
        llm_run.end_time = datetime.now(timezone.utc)
        llm_run.events.append({"name": "error", "time": llm_run.end_time})
        self._end_trace(llm_run)
        self._on_llm_error(llm_run)
        return llm_run

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        tags: Optional[List[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_type: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Run:
        """Start a trace for a chain run."""
        parent_run_id_ = str(parent_run_id) if parent_run_id else None
        execution_order = self._get_execution_order(parent_run_id_)
        start_time = datetime.now(timezone.utc)
        if metadata:
            kwargs.update({"metadata": metadata})
        chain_run = Run(
            id=run_id,
            parent_run_id=parent_run_id,
            serialized=serialized,
            inputs=self._get_chain_inputs(inputs),
            extra=kwargs,
            events=[{"name": "start", "time": start_time}],
            start_time=start_time,
            execution_order=execution_order,
            child_execution_order=execution_order,
            child_runs=[],
            run_type=run_type or "chain",
            name=name,  # type: ignore[arg-type]
            tags=tags or [],
        )
        self._start_trace(chain_run)
        self._on_chain_start(chain_run)
        return chain_run

    def _get_chain_inputs(self, inputs: Any) -> Any:
        """Get the inputs for a chain run."""
        if self._schema_format == "original":
            return inputs if isinstance(inputs, dict) else {"input": inputs}
        elif self._schema_format == "streaming_events":
            return {
                "input": inputs,
            }
        else:
            raise ValueError(f"Invalid format: {self._schema_format}")

    def _get_chain_outputs(self, outputs: Any) -> Any:
        """Get the outputs for a chain run."""
        if self._schema_format == "original":
            return outputs if isinstance(outputs, dict) else {"output": outputs}
        elif self._schema_format == "streaming_events":
            return {
                "output": outputs,
            }
        else:
            raise ValueError(f"Invalid format: {self._schema_format}")

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Run:
        """End a trace for a chain run."""
        chain_run = self._get_run(run_id)
        chain_run.outputs = self._get_chain_outputs(outputs)
        chain_run.end_time = datetime.now(timezone.utc)
        chain_run.events.append({"name": "end", "time": chain_run.end_time})
        if inputs is not None:
            chain_run.inputs = self._get_chain_inputs(inputs)
        self._end_trace(chain_run)
        self._on_chain_end(chain_run)
        return chain_run

    def on_chain_error(
        self,
        error: BaseException,
        *,
        inputs: Optional[Dict[str, Any]] = None,
        run_id: UUID,
        **kwargs: Any,
    ) -> Run:
        """Handle an error for a chain run."""
        chain_run = self._get_run(run_id)
        chain_run.error = self._get_stacktrace(error)
        chain_run.end_time = datetime.now(timezone.utc)
        chain_run.events.append({"name": "error", "time": chain_run.end_time})
        if inputs is not None:
            chain_run.inputs = self._get_chain_inputs(inputs)
        self._end_trace(chain_run)
        self._on_chain_error(chain_run)
        return chain_run

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        tags: Optional[List[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Run:
        """Start a trace for a tool run."""
        parent_run_id_ = str(parent_run_id) if parent_run_id else None
        execution_order = self._get_execution_order(parent_run_id_)
        start_time = datetime.now(timezone.utc)
        if metadata:
            kwargs.update({"metadata": metadata})

        if self._schema_format == "original":
            inputs = {"input": input_str}
        elif self._schema_format == "streaming_events":
            inputs = {"input": inputs}
        else:
            raise AssertionError(f"Invalid format: {self._schema_format}")

        tool_run = Run(
            id=run_id,
            parent_run_id=parent_run_id,
            serialized=serialized,
            # Wrapping in dict since Run requires a dict object.
            inputs=inputs,
            extra=kwargs,
            events=[{"name": "start", "time": start_time}],
            start_time=start_time,
            execution_order=execution_order,
            child_execution_order=execution_order,
            child_runs=[],
            run_type="tool",
            tags=tags or [],
            name=name,  # type: ignore[arg-type]
        )
        self._start_trace(tool_run)
        self._on_tool_start(tool_run)
        return tool_run

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> Run:
        """End a trace for a tool run."""
        tool_run = self._get_run(run_id, run_type="tool")
        tool_run.outputs = {"output": output}
        tool_run.end_time = datetime.now(timezone.utc)
        tool_run.events.append({"name": "end", "time": tool_run.end_time})
        self._end_trace(tool_run)
        self._on_tool_end(tool_run)
        return tool_run

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> Run:
        """Handle an error for a tool run."""
        tool_run = self._get_run(run_id, run_type="tool")
        tool_run.error = self._get_stacktrace(error)
        tool_run.end_time = datetime.now(timezone.utc)
        tool_run.events.append({"name": "error", "time": tool_run.end_time})
        self._end_trace(tool_run)
        self._on_tool_error(tool_run)
        return tool_run

    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Run:
        """Run when Retriever starts running."""
        parent_run_id_ = str(parent_run_id) if parent_run_id else None
        execution_order = self._get_execution_order(parent_run_id_)
        start_time = datetime.now(timezone.utc)
        if metadata:
            kwargs.update({"metadata": metadata})
        retrieval_run = Run(
            id=run_id,
            name=name or "Retriever",
            parent_run_id=parent_run_id,
            serialized=serialized,
            inputs={"query": query},
            extra=kwargs,
            events=[{"name": "start", "time": start_time}],
            start_time=start_time,
            execution_order=execution_order,
            child_execution_order=execution_order,
            tags=tags,
            child_runs=[],
            run_type="retriever",
        )
        self._start_trace(retrieval_run)
        self._on_retriever_start(retrieval_run)
        return retrieval_run

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> Run:
        """Run when Retriever errors."""
        retrieval_run = self._get_run(run_id, run_type="retriever")
        retrieval_run.error = self._get_stacktrace(error)
        retrieval_run.end_time = datetime.now(timezone.utc)
        retrieval_run.events.append({"name": "error", "time": retrieval_run.end_time})
        self._end_trace(retrieval_run)
        self._on_retriever_error(retrieval_run)
        return retrieval_run

    def on_retriever_end(
        self, documents: Sequence[Document], *, run_id: UUID, **kwargs: Any
    ) -> Run:
        """Run when Retriever ends running."""
        retrieval_run = self._get_run(run_id, run_type="retriever")
        retrieval_run.outputs = {"documents": documents}
        retrieval_run.end_time = datetime.now(timezone.utc)
        retrieval_run.events.append({"name": "end", "time": retrieval_run.end_time})
        self._end_trace(retrieval_run)
        self._on_retriever_end(retrieval_run)
        return retrieval_run

    def __deepcopy__(self, memo: dict) -> BaseTracer:
        """Deepcopy the tracer."""
        return self

    def __copy__(self) -> BaseTracer:
        """Copy the tracer."""
        return self

    def _on_run_create(self, run: Run) -> None:
        """Process a run upon creation."""

    def _on_run_update(self, run: Run) -> None:
        """Process a run upon update."""

    def _on_llm_start(self, run: Run) -> None:
        """Process the LLM Run upon start."""

    def _on_llm_new_token(
        self,
        run: Run,
        token: str,
        chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]],
    ) -> None:
        """Process new LLM token."""

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

    def _on_retriever_start(self, run: Run) -> None:
        """Process the Retriever Run upon start."""

    def _on_retriever_end(self, run: Run) -> None:
        """Process the Retriever Run."""

    def _on_retriever_error(self, run: Run) -> None:
        """Process the Retriever Run upon error."""
