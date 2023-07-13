"""A Tracer implementation that records to LangChain endpoint."""
from __future__ import annotations

import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor, wait
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Union
from uuid import UUID

from langsmith import Client

from langchain.callbacks.tracers.base import BaseTracer
from langchain.callbacks.tracers.schemas import Run, RunTypeEnum, TracerSession
from langchain.env import get_runtime_environment
from langchain.load.dump import dumpd
from langchain.schema.messages import BaseMessage

logger = logging.getLogger(__name__)
_LOGGED = set()
_TRACERS: List[LangChainTracer] = []
_CLIENT: Optional[Client] = None


def log_error_once(method: str, exception: Exception) -> None:
    """Log an error once."""
    global _LOGGED
    if (method, type(exception)) in _LOGGED:
        return
    _LOGGED.add((method, type(exception)))
    logger.error(exception)


def wait_for_all_tracers() -> None:
    """Wait for all tracers to finish."""
    global _TRACERS
    for tracer in _TRACERS:
        tracer.wait_for_futures()


def _get_client() -> Client:
    """Get the client."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = Client()
    return _CLIENT


class LangChainTracer(BaseTracer):
    """An implementation of the SharedTracer that POSTS to the langchain endpoint."""

    def __init__(
        self,
        example_id: Optional[Union[UUID, str]] = None,
        project_name: Optional[str] = None,
        client: Optional[Client] = None,
        tags: Optional[List[str]] = None,
        use_threading: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the LangChain tracer."""
        super().__init__(**kwargs)
        self.session: Optional[TracerSession] = None
        self.example_id = (
            UUID(example_id) if isinstance(example_id, str) else example_id
        )
        self.project_name = project_name or os.getenv(
            "LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_SESSION", "default")
        )
        if use_threading:
            # set max_workers to 1 to process tasks in order
            self.executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(
                max_workers=1
            )
        else:
            self.executor = None
        self.client = client or _get_client()
        self._futures: Set[Future] = set()
        self.tags = tags or []
        global _TRACERS
        _TRACERS.append(self)

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        tags: Optional[List[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Start a trace for an LLM run."""
        parent_run_id_ = str(parent_run_id) if parent_run_id else None
        execution_order = self._get_execution_order(parent_run_id_)
        start_time = datetime.utcnow()
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
            run_type=RunTypeEnum.llm,
            tags=tags,
        )
        self._start_trace(chat_model_run)
        self._on_chat_model_start(chat_model_run)

    def _persist_run(self, run: Run) -> None:
        """The Langchain Tracer uses Post/Patch rather than persist."""

    def _get_tags(self, run: Run) -> List[str]:
        """Get combined tags for a run."""
        tags = set(run.tags or [])
        tags.update(self.tags or [])
        return list(tags)

    def _persist_run_single(self, run: Run) -> None:
        """Persist a run."""
        run_dict = run.dict(exclude={"child_runs"})
        run_dict["tags"] = self._get_tags(run)
        extra = run_dict.get("extra", {})
        extra["runtime"] = get_runtime_environment()
        run_dict["extra"] = extra
        try:
            self.client.create_run(**run_dict, project_name=self.project_name)
        except Exception as e:
            # Errors are swallowed by the thread executor so we need to log them here
            log_error_once("post", e)
            raise

    def _update_run_single(self, run: Run) -> None:
        """Update a run."""
        try:
            run_dict = run.dict()
            run_dict["tags"] = self._get_tags(run)
            self.client.update_run(run.id, **run_dict)
        except Exception as e:
            # Errors are swallowed by the thread executor so we need to log them here
            log_error_once("patch", e)
            raise

    def _submit(self, function: Callable[[Run], None], run: Run) -> None:
        """Submit a function to the executor."""
        if self.executor is None:
            function(run)
        else:
            self._futures.add(self.executor.submit(function, run))

    def _on_llm_start(self, run: Run) -> None:
        """Persist an LLM run."""
        if run.parent_run_id is None:
            run.reference_example_id = self.example_id
        self._submit(self._persist_run_single, run.copy(deep=True))

    def _on_chat_model_start(self, run: Run) -> None:
        """Persist an LLM run."""
        if run.parent_run_id is None:
            run.reference_example_id = self.example_id
        self._submit(self._persist_run_single, run.copy(deep=True))

    def _on_llm_end(self, run: Run) -> None:
        """Process the LLM Run."""
        self._submit(self._update_run_single, run.copy(deep=True))

    def _on_llm_error(self, run: Run) -> None:
        """Process the LLM Run upon error."""
        self._submit(self._update_run_single, run.copy(deep=True))

    def _on_chain_start(self, run: Run) -> None:
        """Process the Chain Run upon start."""
        if run.parent_run_id is None:
            run.reference_example_id = self.example_id
        self._submit(self._persist_run_single, run.copy(deep=True))

    def _on_chain_end(self, run: Run) -> None:
        """Process the Chain Run."""
        self._submit(self._update_run_single, run.copy(deep=True))

    def _on_chain_error(self, run: Run) -> None:
        """Process the Chain Run upon error."""
        self._submit(self._update_run_single, run.copy(deep=True))

    def _on_tool_start(self, run: Run) -> None:
        """Process the Tool Run upon start."""
        if run.parent_run_id is None:
            run.reference_example_id = self.example_id
        self._submit(self._persist_run_single, run.copy(deep=True))

    def _on_tool_end(self, run: Run) -> None:
        """Process the Tool Run."""
        self._submit(self._update_run_single, run.copy(deep=True))

    def _on_tool_error(self, run: Run) -> None:
        """Process the Tool Run upon error."""
        self._submit(self._update_run_single, run.copy(deep=True))

    def _on_retriever_start(self, run: Run) -> None:
        """Process the Retriever Run upon start."""
        if run.parent_run_id is None:
            run.reference_example_id = self.example_id
        self._submit(self._persist_run_single, run.copy(deep=True))

    def _on_retriever_end(self, run: Run) -> None:
        """Process the Retriever Run."""
        self._submit(self._update_run_single, run.copy(deep=True))

    def _on_retriever_error(self, run: Run) -> None:
        """Process the Retriever Run upon error."""
        self._submit(self._update_run_single, run.copy(deep=True))

    def wait_for_futures(self) -> None:
        """Wait for the given futures to complete."""
        futures = list(self._futures)
        wait(futures)
        for future in futures:
            self._futures.remove(future)
