"""A Tracer implementation that records to LangChain endpoint."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from uuid import UUID

from langsmith import Client
from langsmith import utils as ls_utils
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from langchain_core.env import get_runtime_environment
from langchain_core.load import dumpd
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)
_LOGGED = set()
_CLIENT: Optional[Client] = None
_EXECUTOR: Optional[ThreadPoolExecutor] = None


def log_error_once(method: str, exception: Exception) -> None:
    """Log an error once.

    Args:
        method: The method that raised the exception.
        exception: The exception that was raised.
    """
    global _LOGGED
    if (method, type(exception)) in _LOGGED:
        return
    _LOGGED.add((method, type(exception)))
    logger.error(exception)


def wait_for_all_tracers() -> None:
    """Wait for all tracers to finish."""
    global _CLIENT
    if _CLIENT is not None and _CLIENT.tracing_queue is not None:
        _CLIENT.tracing_queue.join()


def get_client() -> Client:
    """Get the client."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = Client()
    return _CLIENT


def _get_executor() -> ThreadPoolExecutor:
    """Get the executor."""
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor()
    return _EXECUTOR


def _run_to_dict(run: Run) -> dict:
    return {
        **run.dict(exclude={"child_runs", "inputs", "outputs"}),
        "inputs": run.inputs.copy() if run.inputs is not None else None,
        "outputs": run.outputs.copy() if run.outputs is not None else None,
    }


class LangChainTracer(BaseTracer):
    """Implementation of the SharedTracer that POSTS to the LangChain endpoint."""

    def __init__(
        self,
        example_id: Optional[Union[UUID, str]] = None,
        project_name: Optional[str] = None,
        client: Optional[Client] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the LangChain tracer.

        Args:
            example_id: The example ID.
            project_name: The project name. Defaults to the tracer project.
            client: The client. Defaults to the global client.
            tags: The tags. Defaults to an empty list.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.example_id = (
            UUID(example_id) if isinstance(example_id, str) else example_id
        )
        self.project_name = project_name or ls_utils.get_tracer_project()
        self.client = client or get_client()
        self.tags = tags or []
        self.latest_run: Optional[Run] = None

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
        """Start a trace for an LLM run.

        Args:
            serialized: The serialized model.
            messages: The messages.
            run_id: The run ID.
            tags: The tags. Defaults to None.
            parent_run_id: The parent run ID. Defaults to None.
            metadata: The metadata. Defaults to None.
            name: The name. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Run: The run.
        """
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
            run_type="llm",
            tags=tags,
            name=name,  # type: ignore[arg-type]
        )
        self._start_trace(chat_model_run)
        self._on_chat_model_start(chat_model_run)
        return chat_model_run

    def _persist_run(self, run: Run) -> None:
        run_ = run.copy()
        run_.reference_example_id = self.example_id
        self.latest_run = run_

    def get_run_url(self) -> str:
        """Get the LangSmith root run URL.

        Returns:
            str: The LangSmith root run URL.

        Raises:
            ValueError: If no traced run is found.
            ValueError: If the run URL cannot be found.
        """
        if not self.latest_run:
            raise ValueError("No traced run found.")
        # If this is the first run in a project, the project may not yet be created.
        # This method is only really useful for debugging flows, so we will assume
        # there is some tolerace for latency.
        for attempt in Retrying(
            stop=stop_after_attempt(5),
            wait=wait_exponential_jitter(),
            retry=retry_if_exception_type(ls_utils.LangSmithError),
        ):
            with attempt:
                return self.client.get_run_url(
                    run=self.latest_run, project_name=self.project_name
                )
        raise ValueError("Failed to get run URL.")

    def _get_tags(self, run: Run) -> List[str]:
        """Get combined tags for a run."""
        tags = set(run.tags or [])
        tags.update(self.tags or [])
        return list(tags)

    def _persist_run_single(self, run: Run) -> None:
        """Persist a run."""
        run_dict = _run_to_dict(run)
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
            run_dict = _run_to_dict(run)
            run_dict["tags"] = self._get_tags(run)
            self.client.update_run(run.id, **run_dict)
        except Exception as e:
            # Errors are swallowed by the thread executor so we need to log them here
            log_error_once("patch", e)
            raise

    def _on_llm_start(self, run: Run) -> None:
        """Persist an LLM run."""
        if run.parent_run_id is None:
            run.reference_example_id = self.example_id
        self._persist_run_single(run)

    def _on_chat_model_start(self, run: Run) -> None:
        """Persist an LLM run."""
        if run.parent_run_id is None:
            run.reference_example_id = self.example_id
        self._persist_run_single(run)

    def _on_llm_end(self, run: Run) -> None:
        """Process the LLM Run."""
        self._update_run_single(run)

    def _on_llm_error(self, run: Run) -> None:
        """Process the LLM Run upon error."""
        self._update_run_single(run)

    def _on_chain_start(self, run: Run) -> None:
        """Process the Chain Run upon start."""
        if run.parent_run_id is None:
            run.reference_example_id = self.example_id
        self._persist_run_single(run)

    def _on_chain_end(self, run: Run) -> None:
        """Process the Chain Run."""
        self._update_run_single(run)

    def _on_chain_error(self, run: Run) -> None:
        """Process the Chain Run upon error."""
        self._update_run_single(run)

    def _on_tool_start(self, run: Run) -> None:
        """Process the Tool Run upon start."""
        if run.parent_run_id is None:
            run.reference_example_id = self.example_id
        self._persist_run_single(run)

    def _on_tool_end(self, run: Run) -> None:
        """Process the Tool Run."""
        self._update_run_single(run)

    def _on_tool_error(self, run: Run) -> None:
        """Process the Tool Run upon error."""
        self._update_run_single(run)

    def _on_retriever_start(self, run: Run) -> None:
        """Process the Retriever Run upon start."""
        if run.parent_run_id is None:
            run.reference_example_id = self.example_id
        self._persist_run_single(run)

    def _on_retriever_end(self, run: Run) -> None:
        """Process the Retriever Run."""
        self._update_run_single(run)

    def _on_retriever_error(self, run: Run) -> None:
        """Process the Retriever Run upon error."""
        self._update_run_single(run)

    def wait_for_futures(self) -> None:
        """Wait for the given futures to complete."""
        if self.client is not None and self.client.tracing_queue is not None:
            self.client.tracing_queue.join()
