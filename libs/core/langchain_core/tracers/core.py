"""Utilities for the root listener."""

from __future__ import annotations

import logging
import sys
import traceback
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Optional,
    Union,
    cast,
)

from langchain_core.exceptions import TracerException
from langchain_core.load import dumpd
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    GenerationChunk,
    LLMResult,
)
from langchain_core.tracers.schemas import Run

if TYPE_CHECKING:
    from collections.abc import Coroutine, Sequence
    from uuid import UUID

    from tenacity import RetryCallState

    from langchain_core.documents import Document
    from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

SCHEMA_FORMAT_TYPE = Literal["original", "streaming_events"]


class _TracerCore(ABC):
    """Abstract base class for tracers.

    This class provides common methods, and reusable methods for tracers.
    """

    log_missing_parent: bool = True

    def __init__(
        self,
        *,
        _schema_format: Literal[
            "original", "streaming_events", "original+chat"
        ] = "original",
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
                - 'original+chat' is a format that is the same as 'original'
                  except it does NOT raise an attribute error on_chat_model_start
            kwargs: Additional keyword arguments that will be passed to
                the superclass.
        """
        super().__init__(**kwargs)
        self._schema_format = _schema_format  # For internal use only API will change.
        self.run_map: dict[str, Run] = {}
        """Map of run ID to run. Cleared on run end."""
        self.order_map: dict[UUID, tuple[UUID, str]] = {}
        """Map of run ID to (trace_id, dotted_order). Cleared when tracer GCed."""

    @abstractmethod
    def _persist_run(self, run: Run) -> Union[None, Coroutine[Any, Any, None]]:
        """Persist a run."""

    @staticmethod
    def _add_child_run(
        parent_run: Run,
        child_run: Run,
    ) -> None:
        """Add child run to a chain run or tool run."""
        parent_run.child_runs.append(child_run)

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

    def _start_trace(self, run: Run) -> Union[None, Coroutine[Any, Any, None]]:  # type: ignore[return]
        current_dotted_order = run.start_time.strftime("%Y%m%dT%H%M%S%fZ") + str(run.id)
        if run.parent_run_id:
            if parent := self.order_map.get(run.parent_run_id):
                run.trace_id, run.dotted_order = parent
                run.dotted_order += "." + current_dotted_order
                if parent_run := self.run_map.get(str(run.parent_run_id)):
                    self._add_child_run(parent_run, run)
            else:
                if self.log_missing_parent:
                    logger.debug(
                        "Parent run %s not found for run %s. Treating as a root run.",
                        run.parent_run_id,
                        run.id,
                    )
                run.parent_run_id = None
                run.trace_id = run.id
                run.dotted_order = current_dotted_order
        else:
            run.trace_id = run.id
            run.dotted_order = current_dotted_order
        self.order_map[run.id] = (run.trace_id, run.dotted_order)
        self.run_map[str(run.id)] = run

    def _get_run(
        self, run_id: UUID, run_type: Union[str, set[str], None] = None
    ) -> Run:
        try:
            run = self.run_map[str(run_id)]
        except KeyError as exc:
            msg = f"No indexed run ID {run_id}."
            raise TracerException(msg) from exc

        if isinstance(run_type, str):
            run_types: Union[set[str], None] = {run_type}
        else:
            run_types = run_type
        if run_types is not None and run.run_type not in run_types:
            msg = (
                f"Found {run.run_type} run at ID {run_id}, "
                f"but expected {run_types} run."
            )
            raise TracerException(msg)
        return run

    def _create_chat_model_run(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        run_id: UUID,
        tags: Optional[list[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Run:
        """Create a chat model run."""
        if self._schema_format not in ("streaming_events", "original+chat"):
            # Please keep this un-implemented for backwards compatibility.
            # When it's unimplemented old tracers that use the "original" format
            # fallback on the on_llm_start method implementation if they
            # find that the on_chat_model_start method is not implemented.
            # This can eventually be cleaned up by writing a "modern" tracer
            # that has all the updated schema changes corresponding to
            # the "streaming_events" format.
            msg = (
                f"Chat model tracing is not supported in "
                f"for {self._schema_format} format."
            )
            raise NotImplementedError(msg)
        start_time = datetime.now(timezone.utc)
        if metadata:
            kwargs.update({"metadata": metadata})
        return Run(
            id=run_id,
            parent_run_id=parent_run_id,
            serialized=serialized,
            inputs={"messages": [[dumpd(msg) for msg in batch] for batch in messages]},
            extra=kwargs,
            events=[{"name": "start", "time": start_time}],
            start_time=start_time,
            # WARNING: This is valid ONLY for streaming_events.
            # run_type="llm" is what's used by virtually all tracers.
            # Changing this to "chat_model" may break triggering on_llm_start
            run_type="chat_model",
            tags=tags,
            name=name,  # type: ignore[arg-type]
        )

    def _create_llm_run(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        run_id: UUID,
        tags: Optional[list[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Run:
        """Create a llm run."""
        start_time = datetime.now(timezone.utc)
        if metadata:
            kwargs.update({"metadata": metadata})
        return Run(
            id=run_id,
            parent_run_id=parent_run_id,
            serialized=serialized,
            # TODO: Figure out how to expose kwargs here
            inputs={"prompts": prompts},
            extra=kwargs,
            events=[{"name": "start", "time": start_time}],
            start_time=start_time,
            run_type="llm",
            tags=tags or [],
            name=name,  # type: ignore[arg-type]
        )

    def _llm_run_with_token_event(
        self,
        token: str,
        run_id: UUID,
        chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
        parent_run_id: Optional[UUID] = None,  # noqa: ARG002
    ) -> Run:
        """Append token event to LLM run and return the run."""
        llm_run = self._get_run(run_id, run_type={"llm", "chat_model"})
        event_kwargs: dict[str, Any] = {"token": token}
        if chunk:
            event_kwargs["chunk"] = chunk
        llm_run.events.append(
            {
                "name": "new_token",
                "time": datetime.now(timezone.utc),
                "kwargs": event_kwargs,
            },
        )
        return llm_run

    def _llm_run_with_retry_event(
        self,
        retry_state: RetryCallState,
        run_id: UUID,
    ) -> Run:
        llm_run = self._get_run(run_id)
        retry_d: dict[str, Any] = {
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

    def _complete_llm_run(self, response: LLMResult, run_id: UUID) -> Run:
        llm_run = self._get_run(run_id, run_type={"llm", "chat_model"})
        if getattr(llm_run, "outputs", None) is None:
            llm_run.outputs = {}
        else:
            llm_run.outputs = cast("dict[str, Any]", llm_run.outputs)
        if not llm_run.extra.get("__omit_auto_outputs", False):
            llm_run.outputs.update(response.model_dump())
        for i, generations in enumerate(response.generations):
            for j, generation in enumerate(generations):
                output_generation = llm_run.outputs["generations"][i][j]
                if "message" in output_generation:
                    output_generation["message"] = dumpd(
                        cast("ChatGeneration", generation).message
                    )
        llm_run.end_time = datetime.now(timezone.utc)
        llm_run.events.append({"name": "end", "time": llm_run.end_time})

        return llm_run

    def _errored_llm_run(
        self, error: BaseException, run_id: UUID, response: Optional[LLMResult] = None
    ) -> Run:
        llm_run = self._get_run(run_id, run_type={"llm", "chat_model"})
        llm_run.error = self._get_stacktrace(error)
        if response:
            if getattr(llm_run, "outputs", None) is None:
                llm_run.outputs = {}
            else:
                llm_run.outputs = cast("dict[str, Any]", llm_run.outputs)
            if not llm_run.extra.get("__omit_auto_outputs", False):
                llm_run.outputs.update(response.model_dump())
            for i, generations in enumerate(response.generations):
                for j, generation in enumerate(generations):
                    output_generation = llm_run.outputs["generations"][i][j]
                    if "message" in output_generation:
                        output_generation["message"] = dumpd(
                            cast("ChatGeneration", generation).message
                        )
        llm_run.end_time = datetime.now(timezone.utc)
        llm_run.events.append({"name": "error", "time": llm_run.end_time})

        return llm_run

    def _create_chain_run(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        run_id: UUID,
        tags: Optional[list[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[dict[str, Any]] = None,
        run_type: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Run:
        """Create a chain Run."""
        start_time = datetime.now(timezone.utc)
        if metadata:
            kwargs.update({"metadata": metadata})
        return Run(
            id=run_id,
            parent_run_id=parent_run_id,
            serialized=serialized,
            inputs=self._get_chain_inputs(inputs),
            extra=kwargs,
            events=[{"name": "start", "time": start_time}],
            start_time=start_time,
            child_runs=[],
            run_type=run_type or "chain",
            name=name,  # type: ignore[arg-type]
            tags=tags or [],
        )

    def _get_chain_inputs(self, inputs: Any) -> Any:
        """Get the inputs for a chain run."""
        if self._schema_format in ("original", "original+chat"):
            return inputs if isinstance(inputs, dict) else {"input": inputs}
        if self._schema_format == "streaming_events":
            return {
                "input": inputs,
            }
        msg = f"Invalid format: {self._schema_format}"
        raise ValueError(msg)

    def _get_chain_outputs(self, outputs: Any) -> Any:
        """Get the outputs for a chain run."""
        if self._schema_format in ("original", "original+chat"):
            return outputs if isinstance(outputs, dict) else {"output": outputs}
        if self._schema_format == "streaming_events":
            return {
                "output": outputs,
            }
        msg = f"Invalid format: {self._schema_format}"
        raise ValueError(msg)

    def _complete_chain_run(
        self,
        outputs: dict[str, Any],
        run_id: UUID,
        inputs: Optional[dict[str, Any]] = None,
    ) -> Run:
        """Update a chain run with outputs and end time."""
        chain_run = self._get_run(run_id)
        if getattr(chain_run, "outputs", None) is None:
            chain_run.outputs = {}
        if not chain_run.extra.get("__omit_auto_outputs", False):
            cast("dict[str, Any]", chain_run.outputs).update(
                self._get_chain_outputs(outputs)
            )
        chain_run.end_time = datetime.now(timezone.utc)
        chain_run.events.append({"name": "end", "time": chain_run.end_time})
        if inputs is not None:
            chain_run.inputs = self._get_chain_inputs(inputs)
        return chain_run

    def _errored_chain_run(
        self,
        error: BaseException,
        inputs: Optional[dict[str, Any]],
        run_id: UUID,
    ) -> Run:
        chain_run = self._get_run(run_id)
        chain_run.error = self._get_stacktrace(error)
        chain_run.end_time = datetime.now(timezone.utc)
        chain_run.events.append({"name": "error", "time": chain_run.end_time})
        if inputs is not None:
            chain_run.inputs = self._get_chain_inputs(inputs)
        return chain_run

    def _create_tool_run(
        self,
        serialized: dict[str, Any],
        input_str: str,
        run_id: UUID,
        tags: Optional[list[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
        inputs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Run:
        """Create a tool run."""
        start_time = datetime.now(timezone.utc)
        if metadata:
            kwargs.update({"metadata": metadata})

        if self._schema_format in ("original", "original+chat"):
            inputs = {"input": input_str}
        elif self._schema_format == "streaming_events":
            inputs = {"input": inputs}
        else:
            msg = f"Invalid format: {self._schema_format}"
            raise AssertionError(msg)

        return Run(
            id=run_id,
            parent_run_id=parent_run_id,
            serialized=serialized,
            # Wrapping in dict since Run requires a dict object.
            inputs=inputs,
            extra=kwargs,
            events=[{"name": "start", "time": start_time}],
            start_time=start_time,
            child_runs=[],
            run_type="tool",
            tags=tags or [],
            name=name,  # type: ignore[arg-type]
        )

    def _complete_tool_run(
        self,
        output: dict[str, Any],
        run_id: UUID,
    ) -> Run:
        """Update a tool run with outputs and end time."""
        tool_run = self._get_run(run_id, run_type="tool")
        if getattr(tool_run, "outputs", None) is None:
            tool_run.outputs = {}
        if not tool_run.extra.get("__omit_auto_outputs", False):
            cast("dict[str, Any]", tool_run.outputs).update({"output": output})
        tool_run.end_time = datetime.now(timezone.utc)
        tool_run.events.append({"name": "end", "time": tool_run.end_time})
        return tool_run

    def _errored_tool_run(
        self,
        error: BaseException,
        run_id: UUID,
    ) -> Run:
        """Update a tool run with error and end time."""
        tool_run = self._get_run(run_id, run_type="tool")
        tool_run.error = self._get_stacktrace(error)
        tool_run.end_time = datetime.now(timezone.utc)
        tool_run.events.append({"name": "error", "time": tool_run.end_time})
        return tool_run

    def _create_retrieval_run(
        self,
        serialized: dict[str, Any],
        query: str,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Run:
        """Create a retrieval run."""
        start_time = datetime.now(timezone.utc)
        if metadata:
            kwargs.update({"metadata": metadata})
        return Run(
            id=run_id,
            name=name or "Retriever",
            parent_run_id=parent_run_id,
            serialized=serialized,
            inputs={"query": query},
            extra=kwargs,
            events=[{"name": "start", "time": start_time}],
            start_time=start_time,
            tags=tags,
            child_runs=[],
            run_type="retriever",
        )

    def _complete_retrieval_run(
        self,
        documents: Sequence[Document],
        run_id: UUID,
    ) -> Run:
        """Update a retrieval run with outputs and end time."""
        retrieval_run = self._get_run(run_id, run_type="retriever")
        if getattr(retrieval_run, "outputs", None) is None:
            retrieval_run.outputs = {}
        if not retrieval_run.extra.get("__omit_auto_outputs", False):
            cast("dict[str, Any]", retrieval_run.outputs).update(
                {"documents": documents}
            )
        retrieval_run.end_time = datetime.now(timezone.utc)
        retrieval_run.events.append({"name": "end", "time": retrieval_run.end_time})
        return retrieval_run

    def _errored_retrieval_run(
        self,
        error: BaseException,
        run_id: UUID,
    ) -> Run:
        retrieval_run = self._get_run(run_id, run_type="retriever")
        retrieval_run.error = self._get_stacktrace(error)
        retrieval_run.end_time = datetime.now(timezone.utc)
        retrieval_run.events.append({"name": "error", "time": retrieval_run.end_time})
        return retrieval_run

    def __deepcopy__(self, memo: dict) -> _TracerCore:
        """Deepcopy the tracer."""
        return self

    def __copy__(self) -> _TracerCore:
        """Copy the tracer."""
        return self

    def _end_trace(self, run: Run) -> Union[None, Coroutine[Any, Any, None]]:  # noqa: ARG002
        """End a trace for a run."""
        return None

    def _on_run_create(self, run: Run) -> Union[None, Coroutine[Any, Any, None]]:  # noqa: ARG002
        """Process a run upon creation."""
        return None

    def _on_run_update(self, run: Run) -> Union[None, Coroutine[Any, Any, None]]:  # noqa: ARG002
        """Process a run upon update."""
        return None

    def _on_llm_start(self, run: Run) -> Union[None, Coroutine[Any, Any, None]]:  # noqa: ARG002
        """Process the LLM Run upon start."""
        return None

    def _on_llm_new_token(
        self,
        run: Run,  # noqa: ARG002
        token: str,  # noqa: ARG002
        chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]],  # noqa: ARG002
    ) -> Union[None, Coroutine[Any, Any, None]]:
        """Process new LLM token."""
        return None

    def _on_llm_end(self, run: Run) -> Union[None, Coroutine[Any, Any, None]]:  # noqa: ARG002
        """Process the LLM Run."""
        return None

    def _on_llm_error(self, run: Run) -> Union[None, Coroutine[Any, Any, None]]:  # noqa: ARG002
        """Process the LLM Run upon error."""
        return None

    def _on_chain_start(self, run: Run) -> Union[None, Coroutine[Any, Any, None]]:  # noqa: ARG002
        """Process the Chain Run upon start."""
        return None

    def _on_chain_end(self, run: Run) -> Union[None, Coroutine[Any, Any, None]]:  # noqa: ARG002
        """Process the Chain Run."""
        return None

    def _on_chain_error(self, run: Run) -> Union[None, Coroutine[Any, Any, None]]:  # noqa: ARG002
        """Process the Chain Run upon error."""
        return None

    def _on_tool_start(self, run: Run) -> Union[None, Coroutine[Any, Any, None]]:  # noqa: ARG002
        """Process the Tool Run upon start."""
        return None

    def _on_tool_end(self, run: Run) -> Union[None, Coroutine[Any, Any, None]]:  # noqa: ARG002
        """Process the Tool Run."""
        return None

    def _on_tool_error(self, run: Run) -> Union[None, Coroutine[Any, Any, None]]:  # noqa: ARG002
        """Process the Tool Run upon error."""
        return None

    def _on_chat_model_start(self, run: Run) -> Union[None, Coroutine[Any, Any, None]]:  # noqa: ARG002
        """Process the Chat Model Run upon start."""
        return None

    def _on_retriever_start(self, run: Run) -> Union[None, Coroutine[Any, Any, None]]:  # noqa: ARG002
        """Process the Retriever Run upon start."""
        return None

    def _on_retriever_end(self, run: Run) -> Union[None, Coroutine[Any, Any, None]]:  # noqa: ARG002
        """Process the Retriever Run."""
        return None

    def _on_retriever_error(self, run: Run) -> Union[None, Coroutine[Any, Any, None]]:  # noqa: ARG002
        """Process the Retriever Run upon error."""
        return None
